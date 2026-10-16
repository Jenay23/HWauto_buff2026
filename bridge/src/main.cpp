// hw26_sim_bridge — 把 HWauto_buff2026 接到 Daedalus 模拟器
//
// 话题（与 at_vision_simulator/POWER_RUNE_INTERFACE.md 一致）：
//   订阅 rt/image_raw   (BIMG 二进制: "BIMG"|i64 ns|u32 w|u32 h|RGB888)
//   订阅 rt/gimbal_pose (JSON: {"yaw","pitch","roll"} 弧度)
//   发布 cmd/gimbal     (JSON: {"yaw","pitch","roll","fire","control"})
//
// 设计要点：
//   * 图像用"只保留最新帧"的邮箱 + 工作线程，绝不排队（避免落后越积越多）
//   * 图像线程与姿态线程各自独立 Zenoh session
//   * 支持 --capture 抓一帧存 PNG，--selftest 对静图跑检测（验证模型泛化性）
#include <zenoh.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <fstream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Geometry>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "config_io.hpp"
#include "app_state.hpp"
#include "zenoh_io.hpp"
#include "math_util.hpp"
#include "replay.hpp"
#include "selftest.hpp"
#include "hw26_pipeline.hpp"

// 进程级停止标志：定义在全局作用域（app_state.hpp 里 extern 声明），
// 这样其他模块（replay 等）也能引用。
std::atomic<bool> g_stop{false};

namespace {
void on_sigint(int) { g_stop = true; }

std::string arg_value(int argc, char **argv, const std::string &key, const std::string &def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return def;
}
bool has_flag(int argc, char **argv, const std::string &key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}





}  // namespace

static uint64_t g_vis_n = 0;   // visualize_every 用（显示帧计数）
int main(int argc, char **argv) {
  std::signal(SIGINT, on_sigint);
  const std::string cfg_path = arg_value(argc, argv, "--config", "config/bridge.yaml");
  const std::string capture = arg_value(argc, argv, "--capture", "");
  const std::string selftest = arg_value(argc, argv, "--selftest", "");
  const std::string replay = arg_value(argc, argv, "--replay", "");
  const bool selftest_center = has_flag(argc, argv, "--selftest-center");
  const int max_frames = std::stoi(arg_value(argc, argv, "--frames", "0"));
  const bool no_fire = has_flag(argc, argv, "--no-fire");
  const bool headless = has_flag(argc, argv, "--headless");
  const bool no_filter = has_flag(argc, argv, "--no-filter");
  const std::string conf_ov = arg_value(argc, argv, "--conf", "");
  const std::string mode_ov = arg_value(argc, argv, "--mode", "");
  const std::string color_ov = arg_value(argc, argv, "--color", "");
  const std::string frame_ov = arg_value(argc, argv, "--frame", "");

  BridgeConfig bc;
  try {
    bc = load_config(cfg_path);
  } catch (const std::exception &e) {
    std::cerr << "[bridge] config error: " << e.what() << std::endl;
    std::cerr << "[bridge] usage: " << argv[0]
              << " [--config config/bridge.yaml] [--capture out.png] [--selftest img.png]"
                 " [--frames N] [--no-fire] [--headless]"
              << std::endl;
    return 2;
  }
  if (no_fire) bc.algo.fire_enabled = false;
  if (headless) bc.algo.visualize = false;
  // 命令行覆盖，方便现场试参数不必改文件
  if (!conf_ov.empty()) bc.algo.confidence_threshold = std::stof(conf_ov);
  if (!mode_ov.empty()) bc.algo.buff_mode = std::stoi(mode_ov);
  if (!color_ov.empty()) bc.algo.buff_color = std::stoi(color_ov);
  if (!frame_ov.empty())
    bc.algo.frame = (frame_ov == "gimbal")    ? hw26_bridge::RefFrame::kGimbal
                    : (frame_ov == "chassis") ? hw26_bridge::RefFrame::kChassis
                                              : hw26_bridge::RefFrame::kFace;
  // --no-filter: 关掉颜色 + 激活两道过滤，先保证"检出即可跟踪"
  if (no_filter) {
    bc.algo.filter_color = false;
    bc.algo.filter_activation = false;
  }

  // 检测后端此刻还没判定（要等 Pipeline 构造时看模型的输出形状），所以这里只打路径
  std::cout << "[bridge] config=" << cfg_path << "\n"
            << "[bridge] model=" << bc.algo.model_path
            << "  local_model=" << bc.algo.local_model << "\n" << "  (实际生效 mode=" << bc.algo.local_model << ")" <<
            "  blade_half=" << bc.algo.blade_half_size
            << "/" << bc.algo.blade_half_size_hw26
            << "  muzzle_comp=" << (bc.algo.use_muzzle_offset ? "on" : "off") << "\n"
            << "[bridge] mode=" << bc.algo.buff_mode
            << " (1=small,2=big)  my_color=" << bc.algo.buff_color << " (0=BLUE,1=RED)\n"
            << "[bridge] frame=" << (bc.algo.frame == hw26_bridge::RefFrame::kFace ? "face"
                                     : bc.algo.frame == hw26_bridge::RefFrame::kChassis
                                         ? "chassis"
                                         : "gimbal")
            << "  bullet_speed=" << bc.algo.bullet_speed
            << "  pitch_offset_deg=" << bc.algo.pitch_offset_deg << "\n"
            << "[bridge] 护栏(对齐④桥): slew=" << bc.algo.max_slew_deg_s << "deg/s  smooth="
            << bc.algo.cmd_smooth_alpha << "  aim_step<=" << bc.algo.max_aim_step_px
            << "px  stale<=" << bc.max_image_age_ms << "ms\n"
            << "[bridge] fire=" << (bc.algo.fire_enabled ? "ON" : "OFF")
            << "  err<=" << bc.algo.fire_err_rad << " rad x " << bc.algo.fire_stable_frames
            << " frames  cooldown=" << bc.algo.fire_cooldown << " s" << std::endl;

  // ---- selftest：只跑检测，不需要模拟器 ----
  if (!selftest.empty()) return run_detect_selftest(bc, selftest);

  if (selftest_center) return run_selftest_center(bc);

  // ---- replay：视频回放，不需要模拟器 ----
  if (!replay.empty())
    return run_replay(bc, replay, max_frames, headless, has_flag(argc, argv, "--closed-loop"));

  // ---- capture：抓一帧存盘 ----
  if (!capture.empty()) return run_capture(bc, capture, 60);

  // ---- 正常桥接 ----
  std::unique_ptr<hw26_bridge::Pipeline> pipe;
  try {
    pipe = std::make_unique<hw26_bridge::Pipeline>(bc.algo);
  } catch (const std::exception &e) {
    std::cerr << "[bridge] pipeline init failed: " << e.what() << std::endl;
    return 4;
  }

  // 姿态 session
  z_owned_session_t s_pose, s_img;
  z_owned_config_t c1, c2;
  z_config_default(&c1);
  z_config_default(&c2);
  if (z_open(&s_pose, z_move(c1), NULL) != Z_OK || z_open(&s_img, z_move(c2), NULL) != Z_OK) {
    std::cerr << "[bridge] z_open failed" << std::endl;
    return 5;
  }

  z_owned_closure_sample_t cb_pose, cb_img, cb_kpts, cb_stat;
  z_closure_sample(&cb_pose, on_pose, NULL, NULL);
  z_closure_sample(&cb_img, on_image, NULL, NULL);
  z_closure_sample(&cb_kpts, on_rune_kpts, NULL, NULL);
  z_closure_sample(&cb_stat, on_rune_stat, NULL, NULL);
  z_view_keyexpr_t ke_pose, ke_img, ke_cmd, ke_kpts, ke_stat;
  z_view_keyexpr_from_str(&ke_pose, bc.pose_topic.c_str());
  z_view_keyexpr_from_str(&ke_img, bc.image_topic.c_str());
  z_view_keyexpr_from_str(&ke_cmd, bc.cmd_topic.c_str());
  z_view_keyexpr_from_str(&ke_kpts, bc.rune_kpts_topic.c_str());
  z_view_keyexpr_from_str(&ke_stat, bc.rune_stat_topic.c_str());

  z_owned_subscriber_t sub_pose, sub_img, sub_kpts;
  z_owned_publisher_t pub;
  if (z_declare_subscriber(z_loan(s_pose), &sub_pose, z_loan(ke_pose), z_move(cb_pose), NULL) !=
      Z_OK) {
    std::cerr << "[bridge] subscribe pose failed" << std::endl;
    return 6;
  }
  if (z_declare_subscriber(z_loan(s_img), &sub_img, z_loan(ke_img), z_move(cb_img), NULL) != Z_OK) {
    std::cerr << "[bridge] subscribe image failed" << std::endl;
    return 7;
  }
  if (z_declare_subscriber(z_loan(s_img), &sub_kpts, z_loan(ke_kpts), z_move(cb_kpts),
                           NULL) != Z_OK) {
    // 不致命：模拟器还没发这个 topic 时只是没有真值日志
    std::cerr << "[bridge] subscribe rune_kpts failed (真值日志将为空)" << std::endl;
  }
  z_owned_subscriber_t sub_stat;
  if (z_declare_subscriber(z_loan(s_pose), &sub_stat, z_loan(ke_stat), z_move(cb_stat), NULL) !=
      Z_OK) {
    // 同样不致命：没有 rt/stat 时只是没有命中率日志
    std::cerr << "[bridge] subscribe rt/stat failed (命中率日志将为空)" << std::endl;
  }
  if (z_declare_publisher(z_loan(s_img), &pub, z_loan(ke_cmd), NULL) != Z_OK) {
    std::cerr << "[bridge] declare publisher failed" << std::endl;
    return 8;
  }
  std::cout << "[bridge] subscribed " << bc.pose_topic << " + " << bc.image_topic
            << ", publishing " << bc.cmd_topic << std::endl;

  std::atomic<uint64_t> processed{0};
  std::thread worker([&] {
    std::vector<uint8_t> buf;
    int64_t stamp = 0;
    // 真值日志：每处理一帧写一行 "图片时间戳纳秒,{sim,model}"。
    // 每次运行一个独立文件名（带启动时刻），否则多次跑会互相覆盖。
    {
      // record.sh 会用 HW26_KPTS_LOG 指定与本次 log 同名同标签的路径；
      // 直接手跑没设时，退回带时间戳的默认名（也不会覆盖历史）。
      const char *env_kpts = std::getenv("HW26_KPTS_LOG");
      std::string kname;
      if (env_kpts != nullptr && *env_kpts != '\0') {
        kname = env_kpts;
      } else {
        std::time_t t_now = std::time(nullptr);
        std::tm tm_buf{};
        localtime_r(&t_now, &tm_buf);
        char name_buf[64];
        std::strftime(name_buf, sizeof(name_buf), "hw26_rune_kpts_%Y%m%d_%H%M%S.jsonl",
                      &tm_buf);
        kname = name_buf;
      }
      g_kpts_log.open(kname, std::ios::trunc);
      std::cout << "[bridge] rune-kpts log -> " << kname << std::endl;
      // 命中率日志与 kpts 日志同名：<base>.rtstat.csv（由 rt/stat 回调惰性打开）
      {
        std::lock_guard<std::mutex> lk(g_stat_mu);
        const std::string suf = ".kpts.jsonl";
        if (kname.size() > suf.size() &&
            kname.compare(kname.size() - suf.size(), suf.size(), suf) == 0) {
          g_stat_path = kname.substr(0, kname.size() - suf.size()) + ".rtstat.csv";
        } else {
          g_stat_path = kname + ".rtstat.csv";
        }
      }
    }
    while (!g_stop.load()) {
      if (!g_frame.take(&buf, &stamp)) continue;
      GimbalPose pose;
      {
        std::lock_guard<std::mutex> lk(g_pose_mu);
        pose = g_pose;
      }
      BimgFrame f;
      if (!parse_bimg(buf.data(), buf.size(), &f)) continue;

      const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
      const double age_ms = static_cast<double>(now_ns - f.timestamp_ns) / 1e6;
      const bool fresh = age_ms <= bc.max_image_age_ms;

      auto out = pipe->process(f.rgb, static_cast<int>(f.w), static_cast<int>(f.h),
                               f.timestamp_ns, pose.yaw, pose.pitch, pose.roll);

      if (g_kpts_log.is_open()) {
        std::string kj;
        {
          std::lock_guard<std::mutex> lk(g_kpts_mu);
          kj = g_kpts_json;
        }
        if (kj.empty()) kj = "null";
        // 模型的原始输出：每个检出 cls/conf/中心 + 9 个关键点(展平 18 个数)
        std::string mj = "[";
        for (std::size_t di = 0; di < out.dets.size(); ++di) {
          const auto &d = out.dets[di];
          if (di) mj += ',';
          char hb[96];
          std::snprintf(hb, sizeof(hb),
                        "{\"cls\":%d,\"conf\":%.3f,\"cx\":%.1f,\"cy\":%.1f,\"kpts\":[",
                        d.color * 2 + d.label, d.confidence, d.cx, d.cy);
          mj += hb;
          for (int k = 0; k < 18; ++k) {
            char nb[32];
            std::snprintf(nb, sizeof(nb), "%.1f", d.kpts[k]);
            if (k) mj += ',';
            mj += nb;
          }
          mj += "]}";
        }
        mj += ']';
        g_kpts_log << f.timestamp_ns << ",{\"sim\":" << kj << ",\"model\":" << mj << "}\n";
      }

      const bool control = fresh && out.control;
      const bool fire = bc.algo.fire_enabled && control && out.fire;
      const double yaw = control ? out.yaw : pose.yaw;
      const double pitch = control ? out.pitch : pose.pitch;
      const std::string js = make_cmd_json(yaw, pitch, 0.0, fire, control);
      z_owned_bytes_t b;
      z_bytes_copy_from_buf(&b, reinterpret_cast<const uint8_t *>(js.data()), js.size());
      z_put(z_loan(s_img), z_loan(ke_cmd), z_move(b), NULL);

      if (bc.algo.visualize && !out.debug.empty() &&
                (bc.algo.visualize_every <= 1 || fire ||
                 (++g_vis_n % static_cast<uint64_t>(bc.algo.visualize_every)) == 0)) {
        static bool vis_ok = true;
        if (vis_ok) {
          try {
            cv::imshow("hw26_sim_bridge", out.debug);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) g_stop = true;
          } catch (const std::exception &e) {
            vis_ok = false;
            std::cerr << "[bridge] 无法创建显示窗口（无 GUI/无 DISPLAY），已自动关闭可视化: "
                      << e.what() << std::endl;
            std::cerr << "[bridge] 想看图请在 config/bridge.yaml 里设 visualize: 1 并在有桌面的环境运行"
                      << std::endl;
          }
        }
      }
      // 有检出时打印类别明细（限流），用于诊断"检出但被过滤"
      if (out.n_det > 0) {
        static uint64_t det_print = 0;
        if (det_print++ % static_cast<uint64_t>(std::max(1, bc.algo.console_every)) == 0) {
          std::string ds;
          for (const auto &d : out.dets) {
            char b[96];
            std::snprintf(b, sizeof(b), " [c=%d(%s,%s) conf=%.2f @%.0f,%.0f]", d.color * 2 + d.label,
                          d.color == 0 ? "BLUE" : "RED",
                          d.label == 0 ? "INACTIVE" : "ACTIVE", d.confidence, d.cx, d.cy);
            ds += b;
          }
          std::printf("[det] n=%d sel=%d max_score=%.3f%s\n", out.n_det, out.n_selected,
                      out.max_score, ds.c_str());
          std::fflush(stdout);
        }
      }
      if (out.n_det == 0 && (processed % 60 == 0) && processed > 0) {
        std::printf("[det] n=0  max_score=%.3f  (阈值 %.2f)\n", out.max_score,
                    bc.algo.confidence_threshold);
        std::fflush(stdout);
      }
      const uint64_t k = ++processed;
      // ★ 实时帧率：最近 30 帧的【墙钟】(不是图像时间戳)平均 -> 反映真实处理速度
      static std::vector<int64_t> fps_t;
      fps_t.push_back(now_ns);
      if (fps_t.size() > 31) fps_t.erase(fps_t.begin());
      double fps_now = 0.0;
      if (fps_t.size() >= 11) {
        const double dts = static_cast<double>(fps_t.back() - fps_t.front()) / 1e9;
        if (dts > 1e-6) fps_now = (static_cast<double>(fps_t.size()) - 1.0) / dts;
      }
      if (fire || k % static_cast<uint64_t>(std::max(1, bc.algo.console_every)) == 0) {
        std::printf(
            "[bridge] #%llu age=%.0fms det=%d sel=%d solved=%d q=%d st=%d dis=%.2f "
            "roll=%.3f w=%.3f lead=%.3f lat=%.1fms cmd=(%.2f,%.2f) err=%.4f ctrl=%d fire=%d "
            "nt=%d dlt=%.0f rc=%.2f fps=%.1f hit=%lld/%lld\n",
            static_cast<unsigned long long>(k), age_ms, out.n_det, out.n_selected,
            out.solved ? 1 : 0, out.quality ? 1 : 0, out.status, out.buff_dis, out.roll,
            out.roll_velocity, out.lead_s, out.latency_ms, yaw * 180 / M_PI, pitch * 180 / M_PI,
            out.fire_err, control ? 1 : 0, fire ? 1 : 0, out.dual_targets, out.dual_delta_deg,
            out.rc_dist_m, fps_now, g_stat_hits, g_stat_launch);
        std::fflush(stdout);
      }
      // 组号/身份诊断：每帧一行（有检出时）。判定方法：
      //   p0a 在跳                -> A/B 翻了，绿点其实也在跳
      //   p0a 稳但 sa/sb 在跳      -> 组号映射在变
      //   两者都稳而画面还变        -> 变的是别的东西（顶点顺序 / 绘制）
      if (out.n_det > 0 && (fire || k % static_cast<uint64_t>(std::max(1, bc.algo.console_every)) == 0)) {
        std::printf(
            "[grp] nt=%d ndet=%d nsel=%d nk=%zu p0a=%d sa=%d sb=%d conf=(%.2f,%.2f) "
            "g0=(%.0f,%.0f) g1=(%.0f,%.0f) ra=%.3f rb=%.3f rca=%.2f rcb=%.2f\n",
            out.dual_targets, out.n_det, out.n_selected, out.det_kpts.size(), out.dbg_pose0_is_a,
            out.dbg_src_a, out.dbg_src_b, out.dbg_conf0, out.dbg_conf1, out.dbg_g0x, out.dbg_g0y,
            out.dbg_g1x, out.dbg_g1y, out.dbg_roll_a, out.dbg_roll_b, out.dbg_rca_m,
            out.dbg_rcb_m);
        std::fflush(stdout);
      }

      // 大符专项诊断（限流）：用来定位"黄/品红为什么还在换"。
      //   droll = 本帧两片 roll 与上一帧两片 roll 做【不依赖身份】最优配对后的两个残差。
      //           某一对接近 180 度 => 那片发生了 IPPE 双解翻转（相位翻到对面）。
      //   pick  = 两片各自选了第几个 IPPE 解；若某片的 pick 在 0/1 间来回跳，就是它。
      if (out.dual_targets == 2 && k % 10 == 0) {
        std::printf(
            "[dual] droll=(%.1f,%.1f)deg pick=(%d,%d) dres=%.1fdeg chord=%.3fm "
            "rcd=%.2fm dlt=%.0f cand=(%.0f,%.0f)px | ctr=%d n=%d rms=%.3fm | local n=%d rdy=%d "
            "rms=%.1fdeg span=%.0fdeg w=%.2f LK=%d om=%.3f A=%.3f drop=%d lost=%d hit=%d "
            "theta=%.3f ch=%d rej=%d heal=%d dir=%d\n",
            out.droll_deg[0], out.droll_deg[1], out.sol_pick_t[0], out.sol_pick_t[1],
            out.dual_resid_deg, out.chord_m, out.dual_rcd_m, out.dual_delta_deg, out.cand_px[0],
            out.cand_px[1], out.center_src, out.center_n, out.center_rms_m, out.local_n,
            out.local_ready, out.local_rms_deg, out.local_span_deg, out.local_w, out.local_locked, out.local_omega,
            out.local_amp, out.local_drop, out.local_lost, out.local_hit,
            out.local_theta, out.local_chord, out.local_phys_rej, out.local_phys_heal,
            out.dbg_dir_sgn);
        std::fflush(stdout);
      }
      // 单片时的圆心诊断（大符只检出一片、或小符）：[dual] 行不会打印
      if (bc.algo.center_source == 1 && out.dual_targets < 2 && k % 10 == 0) {
        std::printf("[ctr] src=%d n=%d rms=%.4fm  (单片)\n", out.center_src, out.center_n,
                    out.center_rms_m);
        std::fflush(stdout);
      }
      if (max_frames > 0 && static_cast<int>(k) >= max_frames) {
        g_stop = true;
        break;
      }
    }
  });

  const auto t0 = std::chrono::steady_clock::now();
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    static int warned = 0;
    if (el > 5.0 && g_img_count.load() == 0 && warned < 6) {
      ++warned;
      std::cerr << "[bridge] 还没收到图像，等待中 — 模拟器在跑吗？（订阅 " << bc.image_topic
                << "）" << std::endl;
    }
  }
  g_frame.cv.notify_all();
  worker.join();
  std::cout << "\n[bridge] === SUMMARY ===\n"
            << "  image msgs=" << g_img_count.load() << "  processed=" << processed.load()
            << "  pose msgs=" << g_pose_count.load() << std::endl;
  return 0;
}
