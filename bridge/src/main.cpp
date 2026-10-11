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
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <fstream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Geometry>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hw26_pipeline.hpp"

namespace {

std::atomic<bool> g_stop{false};
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

// ---------------- 配置 ----------------
struct BridgeConfig {
  hw26_bridge::Config algo;
  std::string image_topic = "rt/image_raw";
  std::string pose_topic = "rt/gimbal_pose";
  std::string cmd_topic = "cmd/gimbal";
  double max_image_age_ms = 300.0;  // 超过则不发控制（保持云台）
};


// ---- 合成自检：用已知的合成符验证"检测->PnP->瞄准->云台角"整条链的符号 ----
// 造一个符：车体系下 符心 C、面法向 n、扇叶在面内绕 n 转。把扇叶四角按
// 模拟器的投影模型算成像素喂进管道，再和解析解比较。
int run_selftest_aim(const BridgeConfig &bc) {
  using Eigen::Vector3f;
  using Eigen::Matrix3f;
  const auto Rcg = [](double yaw, double pitch, double roll) {
    return Eigen::AngleAxisf(static_cast<float>(yaw), Vector3f::UnitZ()).toRotationMatrix() *
           Eigen::AngleAxisf(static_cast<float>(pitch), Vector3f::UnitY()).toRotationMatrix() *
           Eigen::AngleAxisf(static_cast<float>(roll), Vector3f::UnitX()).toRotationMatrix();
  };
  // 云台系 -> 相机系：必须与管道内部用同一个来源（hw26_pipeline.hpp 的
  // gimbal_to_camera / R_camera2gimbal）。这里**故意不再手写公式**：
  // 若自检与管道用了不同的相机约定，自检会立刻报超差，正是我们要的交叉检查。
  const auto g2c = [](const Vector3f &g) { return hw26_bridge::gimbal_to_camera(g); };
  const double fx = bc.algo.camera_matrix.at<double>(0, 0);
  const double fy = bc.algo.camera_matrix.at<double>(1, 1);
  const double ccx = bc.algo.camera_matrix.at<double>(0, 2);
  const double ccy = bc.algo.camera_matrix.at<double>(1, 2);
  const float h = static_cast<float>(bc.algo.blade_half_size);
  const double pitch_off = bc.algo.pitch_offset_deg * M_PI / 180.0;  // = -0.349
  const double v0 = bc.algo.bullet_speed, gg = bc.algo.gravity;
  const Vector3f tm(static_cast<float>(bc.algo.muzzle_offset[0]),
                    static_cast<float>(bc.algo.muzzle_offset[1]),
                    static_cast<float>(bc.algo.muzzle_offset[2]));

  // 符：符心在车体系，面法向朝机器人
  const Vector3f C(7.0f, 2.5f, 1.20f);
  Vector3f n(-7.0f, -2.5f, 0.6f);
  n.normalize();
  Vector3f u0 = Vector3f::UnitZ() - n * n.z();
  u0.normalize();
  const Vector3f u1 = n.cross(u0);

  std::printf("\n[aim-selftest] 符心=(%.2f,%.2f,%.2f) 面法向=(%.2f,%.2f,%.2f) h=%.3f\n",
              C.x(), C.y(), C.z(), n.x(), n.y(), n.z(), h);
  std::printf("  期望: yaw 与解析解一致; pitch_cmd = 0.349 - 弹道仰角\n\n");
  std::printf("%-22s %-24s %10s %10s %9s %9s %8s\n", "云台(yaw,pitch)deg", "扇叶相位deg",
              "期望yaw", "原始yaw", "期望pitch", "原始pitch", "误差deg");
  int bad = 0, tot = 0;
  for (double gy : {-20.0, 0.0, 20.0}) {
    for (double gp : {-10.0, 0.0, 10.0}) {
      for (double phase : {0.0, 90.0, 200.0, 300.0}) {
        const double gyaw = gy * M_PI / 180.0, gpit = gp * M_PI / 180.0;
        const Matrix3f Rc = Rcg(gyaw, gpit, 0.0);
        // 扇叶方向：绕 n 转 phase
        const double ph = phase * M_PI / 180.0;
        Vector3f zr = std::cos(ph) * u0 + std::sin(ph) * u1;
        zr.normalize();
        const Vector3f yr = zr.cross(n);
        Matrix3f Rr;
        Rr.col(0) = n; Rr.col(1) = yr; Rr.col(2) = zr;
        const Vector3f tb = C + 0.7f * zr;   // 扇叶中心（车体系）
        // 四角（相对扇叶中心），顺序 上/左/下/右
        const Vector3f obj[4] = {{0, 0, h}, {0, h, 0}, {0, 0, -h}, {0, -h, 0}};
        std::vector<cv::Point2f> kpt;
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
          const Vector3f pc = tb + Rr * obj[i];
          const Vector3f pg = Rc.transpose() * pc;
          const Vector3f q = g2c(pg);
          if (q.z() < 0.2f) { ok = false; break; }
          kpt.emplace_back(static_cast<float>(ccx + fx * q.x() / q.z()),
                           static_cast<float>(ccy + fy * q.y() / q.z()));
        }
        if (!ok) continue;
        // kpt[4]=扇叶中心, kpt[5]=臂上一点（用符心方向外推，够用）
        {
          const Vector3f q4 = g2c(Rc.transpose() * tb);
          const Vector3f qc = g2c(Rc.transpose() * C);
          kpt.emplace_back(static_cast<float>(ccx + fx * q4.x() / q4.z()),
                           static_cast<float>(ccy + fy * q4.y() / q4.z()));
          kpt.emplace_back(static_cast<float>(ccx + fx * qc.x() / qc.z()),
                           static_cast<float>(ccy + fy * qc.y() / qc.z()));
        }
        // 期望
        const Vector3f pm = tb - Rc * tm;
        const double exp_yaw = std::atan2(static_cast<double>(pm.y()), static_cast<double>(pm.x()));
        const double d = std::hypot(static_cast<double>(pm.x()), static_cast<double>(pm.y()));
        const double hh = static_cast<double>(pm.z());
        const double v2 = v0 * v0;
        const double disc = v2 * v2 - gg * (gg * d * d + 2.0 * hh * v2);
        const double elev = std::atan((v2 - std::sqrt(std::max(0.0, disc))) / (gg * d));
        const double exp_pitch = -(elev + pitch_off);

        // 喂若干帧让它进 TRACKING
        hw26_bridge::Pipeline pipe(bc.algo);
        pipe.set_external_kpts(kpt);
        cv::Mat img(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
        hw26_bridge::Output out;
        for (int f = 0; f < 200; ++f) {
          out = pipe.process(img.data, 1440, 1080, static_cast<int64_t>(f) * 33333333LL,
                             static_cast<float>(gyaw), static_cast<float>(gpit), 0.f);
        }
        // 用【原始弹道解】验证数学与符号（不受护栏限速影响）
        const double dy = (out.fire_aim_yaw - exp_yaw) * 180.0 / M_PI;
        const double dp = (out.fire_aim_pitch - exp_pitch) * 180.0 / M_PI;
        const double err = std::hypot(dy, dp);
        ++tot;
        if (err > 2.0) ++bad;
        std::printf("%-22s %-24.0f %9.2f° %9.2f° %8.2f° %8.2f° %7.2f %s\n",
                    (std::to_string(static_cast<int>(gy)) + "," +
                     std::to_string(static_cast<int>(gp))).c_str(), phase,
                    exp_yaw * 180 / M_PI, out.fire_aim_yaw * 180 / M_PI, exp_pitch * 180 / M_PI,
                    out.fire_aim_pitch * 180 / M_PI, err, err > 2.0 ? "  <== 超差" : "");
      }
    }
  }
  std::printf("\n[aim-selftest] %d/%d 个用例误差 <= 2deg\n", tot - bad, tot);
  return bad == 0 ? 0 : 5;
}

static double wrap_pi_local(double x) {
  while (x > M_PI) x -= 2 * M_PI;
  while (x < -M_PI) x += 2 * M_PI;
  return x;
}

// ---- 极简扁平行配置解析：key: value / key: [a, b, c] / # 注释 ----
// 不用 cv::FileStorage，避免它对 YAML 方言的挑剔，配置也更好手改。
struct FlatConfig {
  std::map<std::string, std::string> kv;
  std::map<std::string, std::vector<double>> lists;

  bool has(const std::string &k) const { return kv.count(k) || lists.count(k); }
  int geti(const std::string &k, int d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    try { return std::stoi(it->second); } catch (...) { return d; }
  }
  double getd(const std::string &k, double d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    try { return std::stod(it->second); } catch (...) { return d; }
  }
  bool getb(const std::string &k, bool d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    const std::string v = it->second;
    return !(v == "0" || v == "false" || v == "no" || v == "off");
  }
  std::string gets(const std::string &k, const std::string &d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
};

std::string trim(const std::string &s) {
  const size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

FlatConfig parse_flat(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  FlatConfig fc;
  std::string line;
  while (std::getline(in, line)) {
    const size_t h = line.find('#');
    if (h != std::string::npos) line = line.substr(0, h);
    line = trim(line);
    if (line.empty()) continue;
    const size_t c = line.find(':');
    if (c == std::string::npos) continue;
    const std::string key = trim(line.substr(0, c));
    std::string val = trim(line.substr(c + 1));
    if (key.empty()) continue;
    if (val.size() >= 2 && val.front() == '[' && val.back() == ']') {
      std::vector<double> nums;
      std::string cur;
      for (size_t i = 1; i + 1 < val.size(); ++i) {
        const char ch = val[i];
        if (ch == ',') {
          if (!trim(cur).empty()) nums.push_back(std::stod(trim(cur)));
          cur.clear();
        } else {
          cur.push_back(ch);
        }
      }
      if (!trim(cur).empty()) nums.push_back(std::stod(trim(cur)));
      fc.lists[key] = nums;
    } else {
      if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
        val = val.substr(1, val.size() - 2);
      }
      fc.kv[key] = val;
    }
  }
  return fc;
}

BridgeConfig load_config(const std::string &path) {
  BridgeConfig bc;
  const FlatConfig fs = parse_flat(path);

  // 检测后端不再从配置读：只写一个 model: 路径，Pipeline 按模型的输出形状判定
  // （[1,5040,26]->HW26, [1,17,8400]->sp25），因此不可能出现"开关与路径不匹配"。
  bc.algo.model_path = fs.gets("model", bc.algo.model_path);
  bc.algo.blade_half_size = fs.getd("blade_half_size", bc.algo.blade_half_size);
  bc.algo.blade_half_size_hw26 =
      fs.getd("blade_half_size_hw26", bc.algo.blade_half_size_hw26);
  bc.algo.use_muzzle_offset = fs.getb("use_muzzle_offset", bc.algo.use_muzzle_offset);
  const auto mo = fs.lists.count("muzzle_offset") ? fs.lists.at("muzzle_offset") : std::vector<double>{};
  if (mo.size() == 3) {
    bc.algo.muzzle_offset[0] = mo[0];
    bc.algo.muzzle_offset[1] = mo[1];
    bc.algo.muzzle_offset[2] = mo[2];
  }
  if (bc.algo.model_path.empty()) throw std::runtime_error("config: model path is required");
  // 相对路径按“配置文件所在目录”解析，避免依赖 CWD
  if (bc.algo.model_path.front() != '/') {
    const size_t slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
    bc.algo.model_path = dir + "/" + bc.algo.model_path;
  }

  const auto km = fs.lists.count("camera_matrix") ? fs.lists.at("camera_matrix")
                                                  : std::vector<double>{};
  if (km.size() == 9) {
    bc.algo.camera_matrix = (cv::Mat_<double>(3, 3) << km[0], km[1], km[2], km[3], km[4], km[5],
                             km[6], km[7], km[8]);
  }
  const auto dc = fs.lists.count("distort_coeffs") ? fs.lists.at("distort_coeffs")
                                                   : std::vector<double>{};
  if (dc.size() == 5) {
    bc.algo.distort_coeffs = (cv::Mat_<double>(1, 5) << dc[0], dc[1], dc[2], dc[3], dc[4]);
  }

  bc.algo.buff_mode = fs.geti("mode", 1);
  bc.algo.buff_color = fs.geti("my_color", 0);
  bc.algo.confidence_threshold = static_cast<float>(fs.getd("confidence_threshold", 0.5));
  bc.algo.nms_threshold = static_cast<float>(fs.getd("nms_threshold", 0.4));
  bc.algo.filter_color = fs.getb("filter_color", true);
  bc.algo.filter_activation = fs.getb("filter_activation", true);
  bc.algo.dual_target = fs.getb("dual_target", true);
  bc.algo.bullet_speed = fs.getd("bullet_speed", 22.0);
  bc.algo.gravity = fs.getd("gravity", 9.81);
  bc.algo.yaw_offset_deg = fs.getd("yaw_offset_deg", 0.0);
  bc.algo.pitch_offset_deg = fs.getd("pitch_offset_deg", -20.0);
  bc.algo.predict_time = fs.getd("predict_time", 0.10);
  bc.algo.gimbal_lead_time = fs.getd("gimbal_lead_time", 0.0);
  bc.algo.alg_latency = fs.getd("alg_latency", 0.0);
  bc.algo.min_distance = fs.getd("min_distance", 2.0);
  bc.algo.max_distance = fs.getd("max_distance", 15.0);
  bc.algo.fire_err_rad = fs.getd("fire_err_rad", 0.02);
  bc.algo.fire_stable_frames = fs.geti("fire_stable_frames", 3);
  bc.algo.fire_cooldown = fs.getd("fire_cooldown", 1.0);
  bc.algo.fire_enabled = fs.getb("fire_enabled", true);
  bc.algo.visualize = fs.getb("visualize", false);
  bc.algo.log_every = fs.geti("log_every", 30);
  bc.algo.log_path = fs.gets("log_path", "hw26_bridge_log.csv");
  {
    const std::string fr = fs.gets("frame", "face");
    bc.algo.frame = (fr == "gimbal")    ? hw26_bridge::RefFrame::kGimbal
                    : (fr == "chassis") ? hw26_bridge::RefFrame::kChassis
                                        : hw26_bridge::RefFrame::kFace;
  }
  bc.algo.face_ema = fs.getd("face_ema", bc.algo.face_ema);
  bc.algo.face_ema_locked = fs.getd("face_ema_locked", bc.algo.face_ema_locked);
  bc.algo.face_lock_frames = fs.geti("face_lock_frames", bc.algo.face_lock_frames);
  bc.algo.max_cmd_step_deg = fs.getd("max_cmd_step_deg", bc.algo.max_cmd_step_deg);
  bc.algo.max_slew_deg_s = fs.getd("max_slew_deg_s", bc.algo.max_slew_deg_s);
  bc.algo.cmd_smooth_alpha = fs.getd("cmd_smooth_alpha", bc.algo.cmd_smooth_alpha);
  bc.algo.max_aim_step_px = fs.getd("max_aim_step_px", bc.algo.max_aim_step_px);
  bc.algo.stale_ms = fs.getd("stale_ms", bc.algo.stale_ms);
  // 与 ④ 的 stale 门统一：取两者较小者，避免出现两个不一致的过期判据
  bc.algo.refine_r_center = fs.getb("refine_r_center", bc.algo.refine_r_center);
  bc.image_topic = fs.gets("image_topic", bc.image_topic);
  bc.pose_topic = fs.gets("pose_topic", bc.pose_topic);
  bc.cmd_topic = fs.gets("cmd_topic", bc.cmd_topic);
  bc.max_image_age_ms = fs.getd("max_image_age_ms", 300.0);
  // 与 ④ 的 stale 门统一：取较小者，避免两个不一致的过期判据
  bc.max_image_age_ms = std::min(bc.max_image_age_ms, bc.algo.stale_ms);
  return bc;
}

// ---------------- JSON 小工具（自己拼，避免引第三方） ----------------
struct GimbalPose {
  float yaw = 0, pitch = 0, roll = 0;
};

bool parse_pose(const uint8_t *d, size_t n, GimbalPose *out) {
  std::string s(reinterpret_cast<const char *>(d), n);
  auto grab = [&](const char *key, float *dst) -> bool {
    const std::string k = std::string("\"") + key + "\"";
    const size_t p = s.find(k);
    if (p == std::string::npos) return false;
    const size_t c = s.find(':', p + k.size());
    if (c == std::string::npos) return false;
    *dst = std::strtof(s.c_str() + c + 1, nullptr);
    return true;
  };
  bool any = false;
  any |= grab("yaw", &out->yaw);
  any |= grab("pitch", &out->pitch);
  any |= grab("roll", &out->roll);
  return any;
}

std::string make_cmd_json(double yaw, double pitch, double roll, bool fire, bool control) {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "{\"yaw\":%.6f,\"pitch\":%.6f,\"roll\":%.6f,\"fire\":%s,\"control\":%s}", yaw,
                pitch, roll, fire ? "true" : "false", control ? "true" : "false");
  return std::string(buf);
}

// ---------------- BIMG 解析 ----------------
struct BimgFrame {
  int64_t timestamp_ns = 0;
  uint32_t w = 0, h = 0;
  const uint8_t *rgb = nullptr;
  size_t rgb_len = 0;
};

bool parse_bimg(const uint8_t *d, size_t n, BimgFrame *f) {
  if (n < 20 || std::memcmp(d, "BIMG", 4) != 0) return false;
  std::memcpy(&f->timestamp_ns, d + 4, 8);
  std::memcpy(&f->w, d + 12, 4);
  std::memcpy(&f->h, d + 16, 4);
  f->rgb = d + 20;
  f->rgb_len = n - 20;
  return f->rgb_len == static_cast<size_t>(f->w) * f->h * 3;
}

// ---------------- 最新帧邮箱 ----------------
struct LatestFrame {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<uint8_t> data;
  int64_t stamp = 0;
  uint64_t seq = 0;
  uint64_t consumed = 0;
  bool fresh = false;

  void put(const uint8_t *d, size_t n, int64_t ts) {
    std::lock_guard<std::mutex> lk(mu);
    data.assign(d, d + n);
    stamp = ts;
    ++seq;
    fresh = true;
    cv.notify_one();
  }
  bool take(std::vector<uint8_t> *out, int64_t *ts) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait_for(lk, std::chrono::milliseconds(200),
                [&] { return fresh || g_stop.load(); });
    if (!fresh) return false;
    *out = std::move(data);
    *ts = stamp;
    fresh = false;
    consumed = seq;
    return true;
  }
};

LatestFrame g_frame;
std::mutex g_pose_mu;
GimbalPose g_pose;
std::atomic<bool> g_have_pose{false};
std::atomic<uint64_t> g_pose_count{0};
std::atomic<uint64_t> g_img_count{0};
std::atomic<uint64_t> g_dropped{0};
std::string g_capture_path;

void on_pose(struct z_loaned_sample_t *sample, void *) {
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const uint8_t *d = z_slice_data(z_loan(owned));
  const size_t n = z_slice_len(z_loan(owned));
  GimbalPose p;
  if (parse_pose(d, n, &p)) {
    std::lock_guard<std::mutex> lk(g_pose_mu);
    g_pose = p;
    g_have_pose = true;
    g_pose_count++;
  }
  z_drop(z_move(owned));
}

void on_image(struct z_loaned_sample_t *sample, void *) {
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const uint8_t *d = z_slice_data(z_loan(owned));
  const size_t n = z_slice_len(z_loan(owned));

  if (!g_capture_path.empty()) {
    BimgFrame f;
    if (parse_bimg(d, n, &f)) {
      cv::Mat rgb(f.h, f.w, CV_8UC3, const_cast<uint8_t *>(f.rgb));
      cv::Mat bgr;
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      if (cv::imwrite(g_capture_path, bgr)) {
        std::cout << "[bridge] captured " << f.w << "x" << f.h << " -> " << g_capture_path
                  << " (ts=" << f.timestamp_ns << ")" << std::endl;
      } else {
        std::cerr << "[bridge] imwrite failed: " << g_capture_path << std::endl;
      }
      g_stop = true;
    }
  } else {
    BimgFrame f;
    if (!parse_bimg(d, n, &f)) {
      std::cerr << "[bridge] bad BIMG frame (len=" << n << ")" << std::endl;
    } else {
      g_img_count++;
      g_frame.put(d, n, f.timestamp_ns);
    }
  }
  z_drop(z_move(owned));
}

int run_capture(const BridgeConfig &bc, const std::string &out_path, int timeout_s) {
  g_capture_path = out_path;
  z_owned_config_t cfg;
  z_config_default(&cfg);
  z_owned_session_t session;
  if (z_open(&session, z_move(cfg), NULL) != Z_OK) {
    std::cerr << "[bridge] z_open failed" << std::endl;
    return 1;
  }
  z_owned_closure_sample_t cb;
  z_closure_sample(&cb, on_image, NULL, NULL);
  z_view_keyexpr_t ke;
  z_view_keyexpr_from_str(&ke, bc.image_topic.c_str());
  z_owned_subscriber_t sub;
  if (z_declare_subscriber(z_loan(session), &sub, z_loan(ke), z_move(cb), NULL) != Z_OK) {
    std::cerr << "[bridge] subscribe failed: " << bc.image_topic << std::endl;
    return 2;
  }
  std::cout << "[bridge] subscribed " << bc.image_topic << ", waiting for one frame ..."
            << std::endl;
  const auto t0 = std::chrono::steady_clock::now();
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
      std::cerr << "[bridge] timeout waiting for frame (simulator running?)" << std::endl;
      return 3;
    }
  }
  return 0;
}

// ---- 视频回放：不需要模拟器，直接看窗口 + 打通算法链 ----
int run_replay(const BridgeConfig &bc, const std::string &video_path, int max_frames,
               bool headless, bool closed_loop) {
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "[bridge] 打不开视频: " << video_path << std::endl;
    return 3;
  }
  const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (!(fps > 1.0)) fps = 30.0;

  std::cout << "[bridge] replay " << video_path << "  " << W << "x" << H << " @" << fps
            << "fps" << std::endl;
  std::cout << "[bridge] 提示：回放模式假定相机静止，建议 config 里 frame: \"gimbal\"" << std::endl;

  hw26_bridge::Pipeline pipe(bc.algo);
  cv::Mat frame, rgb;
  int n = 0, det_ok = 0, sel_ok = 0, track_ok = 0, fired = 0, sol1 = 0;
  std::vector<double> jump_hist;

  // 闭环模式：模拟模拟器的云台伺服（18 rad/s，pitch 限位 ±0.785），
  // 用它把我们的指令变成下一帧的 gimbal_pose —— 这正是抖动/自激的发生条件
  const double servo_dt = 1.0 / fps;
  const double servo_rate = 18.0;
  const double pitch_lim = 0.785;
  double g_yaw = 0.0, g_pitch = 0.0;
  std::vector<double> cmd_dyaw, cmd_dpitch, cmd_yaw_hist, cmd_pitch_hist;
  double prev_yaw = 0.0, prev_pitch = 0.0;
  bool has_prev_cmd = false;
  double gimbal_travel = 0.0;   // 虚拟云台累计转角（跑飞指标）
  int sat_frames = 0, cl_frames = 0;  // 撞限位的帧数
  const auto wall0 = std::chrono::steady_clock::now();
  while (cap.read(frame) && !g_stop.load()) {
    if (frame.cols != W || frame.rows != H) {
      cv::resize(frame, frame, {W, H});
    }
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    const int64_t ts_ns = static_cast<int64_t>(cap.get(cv::CAP_PROP_POS_MSEC) * 1e6);
    // 回放：相机静止 -> 云台姿态给 0
    auto out = pipe.process(rgb.data, W, H, ts_ns, static_cast<float>(g_yaw),
                            static_cast<float>(g_pitch), 0.f);
    if (out.n_det > 0) ++det_ok;
    if (out.cmd_jump_deg > 0.0) jump_hist.push_back(out.cmd_jump_deg);
    if (out.n_solutions >= 2 && out.sol_pick == 1) ++sol1;
    if (out.n_selected > 0) ++sel_ok;
    if (out.status == 1) ++track_ok;
    if (out.fire) ++fired;
    ++n;

    if (out.n_det > 0) {
      static uint64_t dp = 0;
      if (dp++ % 5 == 0) {
        std::string ds;
        for (const auto &d : out.dets) {
          char b[96];
          std::snprintf(b, sizeof(b), " [c=%d(%s,%s) conf=%.2f @%.0f,%.0f]", d.color * 2 + d.label,
                        d.color == 0 ? "BLUE" : "RED", d.label == 0 ? "INACTIVE" : "ACTIVE",
                        d.confidence, d.cx, d.cy);
          ds += b;
        }
        std::printf("[det] n=%d sel=%d max_score=%.3f%s\n", out.n_det, out.n_selected,
                    out.max_score, ds.c_str());
      }
    }
    if (out.n_det == 0 && n % 60 == 0 && n > 0) {
      std::printf("[det] n=0  max_score=%.3f  (阈值 %.2f)\n", out.max_score,
                  bc.algo.confidence_threshold);
    }
    if (n % 20 == 0 || out.fire) {
      std::printf("[replay] #%d det=%d sel=%d solved=%d q=%d st=%d dis=%.2f roll=%.3f w=%.3f "
                  "aim=(%d,%d) cmd=(%.2f,%.2f) err=%.4f fire=%d ex=%.4fm tilt=%.1fdeg "
                  "G=(%d,%d) B=(%d,%d) reproj=%.2fpx\n",
                  n, out.n_det, out.n_selected, out.solved ? 1 : 0, out.quality ? 1 : 0,
                  out.status, out.buff_dis, out.roll, out.roll_velocity, out.aim_px_x,
                  out.aim_px_y, out.yaw * 180 / M_PI, out.pitch * 180 / M_PI, out.fire_err,
                  out.fire ? 1 : 0, out.face_ex, out.face_tilt_deg, out.green_px_x, out.green_px_y,
                  out.blue_px_x, out.blue_px_y, out.reproj_px);
      std::fflush(stdout);
    }
    if (!headless && !out.debug.empty()) {
      static bool vis_ok = true;
      if (vis_ok) {
        try {
          cv::imshow("hw26_sim_bridge (replay)", out.debug);
          const int key = cv::waitKey(1);
          if (key == 'q' || key == 27) break;
        } catch (const std::exception &e) {
          vis_ok = false;
          std::cerr << "[bridge] 无法创建窗口（无 GUI）：" << e.what() << std::endl;
          std::cerr << "[bridge] 用 --headless 可静默跑完" << std::endl;
        }
      }
    }
    // 闭环：把指令喂给虚拟云台（带 18 rad/s 速率限制）
    if (closed_loop && out.control) {
      const double dy = wrap_pi_local(out.yaw - g_yaw);
      const double dp = out.pitch - g_pitch;
      const double need = std::hypot(dy, dp);
      const double maxstep = servo_rate * servo_dt;
      const double f = (need > maxstep) ? (maxstep / need) : 1.0;
      const double pg = g_pitch;
      g_yaw = wrap_pi_local(g_yaw + dy * f);
      g_pitch = std::clamp(g_pitch + dp * f, -pitch_lim, pitch_lim);
      gimbal_travel += std::hypot(dy * f, g_pitch - pg);
      ++cl_frames;
      if (std::abs(g_pitch) >= pitch_lim - 1e-3) ++sat_frames;
      if (has_prev_cmd) {
        cmd_dyaw.push_back(std::abs(wrap_pi_local(out.yaw - prev_yaw)));
        cmd_dpitch.push_back(std::abs(out.pitch - prev_pitch));
      }
      cmd_yaw_hist.push_back(out.yaw);
      cmd_pitch_hist.push_back(out.pitch);
      has_prev_cmd = true;
    }
    prev_yaw = out.yaw;
    prev_pitch = out.pitch;
    if (max_frames > 0 && n >= max_frames) break;
  }
  auto stats = [](const std::vector<double> &v, double *mean, double *sd, double *mx) {
    if (v.empty()) { *mean = *sd = *mx = 0; return; }
    double s = 0, m = 0;
    for (double x : v) { s += x; m = std::max(m, x); }
    const double mu = s / v.size();
    double var = 0;
    for (double x : v) var += (x - mu) * (x - mu);
    *mean = mu; *sd = std::sqrt(var / v.size()); *mx = m;
  };
  if (!cmd_yaw_hist.empty() && cmd_yaw_hist.size() > 4) {
    std::vector<double> d;
    for (std::size_t i = 1; i < cmd_yaw_hist.size(); ++i)
      d.push_back(wrap_pi_local(cmd_yaw_hist[i] - cmd_yaw_hist[i - 1]));
    double mu = 0;
    for (double x : d) mu += x;
    mu /= d.size();
    double v0 = 0, v1 = 0;
    for (std::size_t i = 0; i < d.size(); ++i) v0 += (d[i] - mu) * (d[i] - mu);
    for (std::size_t i = 1; i < d.size(); ++i) v1 += (d[i] - mu) * (d[i - 1] - mu);
    int flips = 0, longest = 0, run = 0, sign = 0;
    for (double x : d) {
      const int sg = (x > 0) ? 1 : ((x < 0) ? -1 : 0);
      if (sg != 0 && sg == sign) { ++run; } else { if (sg != 0) run = 1; }
      if (sg != 0) {
        if (sign != 0 && sg != sign) ++flips;
        sign = sg;
        longest = std::max(longest, run);
      }
    }
    std::printf("\n[replay] === 指令增量自相关（极限环判据，对齐 sp25 的分析口径）===\n"
                "  yaw 增量 lag-1 自相关 = %+.3f  （接近 -0.5 = 每帧交替 = 极限环）\n"
                "  符号翻转率 = %.1f%%   最长同号连续 = %d 帧\n",
                (v0 > 1e-12) ? (v1 / v0) : 0.0, 100.0 * flips / std::max<std::size_t>(1, d.size() - 1),
                longest);
  }
  if (closed_loop && !cmd_dyaw.empty()) {
    double my, sy, xy, mp, sp, xp;
    stats(cmd_dyaw, &my, &sy, &xy);
    stats(cmd_dpitch, &mp, &sp, &xp);
    std::printf("\n[replay] === 闭环伺服抖动（指令帧间变化量，越小越稳）===\n");
    std::printf("  yaw : 均值 %.4f rad  标准差 %.4f rad  最大 %.4f rad  (%.3f/%.3f/%.3f deg)\n",
                my, sy, xy, my * 180 / M_PI, sy * 180 / M_PI, xy * 180 / M_PI);
    std::printf("  pitch: 均值 %.4f rad  标准差 %.4f rad  最大 %.4f rad  (%.3f/%.3f/%.3f deg)\n",
                mp, sp, xp, mp * 180 / M_PI, sp * 180 / M_PI, xp * 180 / M_PI);
    std::printf("  有效指令帧 %zu\n", cmd_dyaw.size());
    std::printf("  虚拟云台累计转角 %.1f deg（均 %.2f deg/帧）  撞俯仰限位 %d/%d 帧 = %.1f%%\n",
                gimbal_travel * 180 / M_PI, cl_frames ? gimbal_travel / cl_frames * 180 / M_PI : 0.0,
                sat_frames, cl_frames, cl_frames ? 100.0 * sat_frames / cl_frames : 0.0);
  }
  if (!jump_hist.empty()) {
    std::vector<double> j = jump_hist;
    std::sort(j.begin(), j.end());
    auto q = [&](double p) { return j[static_cast<std::size_t>(p * (j.size() - 1))]; };
    std::printf("\n[replay] === 指令跳变(相邻有效指令角度差) ===\n"
                "  n=%zu  p50=%.2f deg  p90=%.2f deg  p99=%.2f deg  max=%.2f deg\n"
                "  >5deg: %.1f%%   >20deg: %.1f%%   >60deg: %.1f%%\n"
                "  IPPE 选到第 2 解的比例: %.1f%%\n",
                j.size(), q(0.50), q(0.90), q(0.99), j.back(),
                100.0 * std::count_if(j.begin(), j.end(), [](double x) { return x > 5; }) / j.size(),
                100.0 * std::count_if(j.begin(), j.end(), [](double x) { return x > 20; }) / j.size(),
                100.0 * std::count_if(j.begin(), j.end(), [](double x) { return x > 60; }) / j.size(),
                100.0 * sol1 / std::max(1, n));
  }
  const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  std::printf("\n[replay] === SUMMARY ===\n  frames=%d  有检出=%d (%.1f%%)  被Selector留下=%d  TRACKING=%d  开火=%d\n"
              "  用时 %.1fs  平均 %.1f ms/帧 (%.1f FPS)\n",
              n, det_ok, n ? 100.0 * det_ok / n : 0.0, sel_ok, track_ok, fired, el,
              n ? el * 1000.0 / n : 0.0, n && el > 0 ? n / el : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, on_sigint);
  const std::string cfg_path = arg_value(argc, argv, "--config", "config/bridge.yaml");
  const std::string capture = arg_value(argc, argv, "--capture", "");
  const std::string selftest = arg_value(argc, argv, "--selftest", "");
  const std::string replay = arg_value(argc, argv, "--replay", "");
  const bool selftest_aim = has_flag(argc, argv, "--selftest-aim");
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
            << "  blade_half=" << bc.algo.blade_half_size
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
  if (!selftest.empty()) {
    cv::Mat img = cv::imread(selftest, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "[bridge] cannot read image: " << selftest << std::endl;
      return 3;
    }
    try {
      hw26_bridge::Pipeline pipe(bc.algo);
      std::cout << "[bridge] selftest on " << selftest << " (" << img.cols << "x" << img.rows
                << ")" << std::endl;
      const int n = pipe.self_test_image(img);
      std::cout << "[bridge] selftest result: " << n << " detection(s)" << std::endl;
      std::cout << (n > 0 ? ">>> 模型对该图有检出" : ">>> 模型对该图无检出（可能不泛化）")
                << std::endl;
      return n > 0 ? 0 : 4;
    } catch (const std::exception &e) {
      std::cerr << "[bridge] selftest failed: " << e.what() << std::endl;
      return 5;
    }
  }

  if (selftest_aim) {
    if (bc.algo.buff_mode == 2) {
      std::cout
          << "[bridge] WARNING: --selftest-aim injects a STATIC blade phase, so the big-rune\n"
             "         tracker can never converge its RANSAC sine fit and will never aim.\n"
             "         Use mode: 1 (small rune) - it exercises exactly the same aim maths.\n";
    }
    return run_selftest_aim(bc);
  }

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

  z_owned_closure_sample_t cb_pose, cb_img;
  z_closure_sample(&cb_pose, on_pose, NULL, NULL);
  z_closure_sample(&cb_img, on_image, NULL, NULL);
  z_view_keyexpr_t ke_pose, ke_img, ke_cmd;
  z_view_keyexpr_from_str(&ke_pose, bc.pose_topic.c_str());
  z_view_keyexpr_from_str(&ke_img, bc.image_topic.c_str());
  z_view_keyexpr_from_str(&ke_cmd, bc.cmd_topic.c_str());

  z_owned_subscriber_t sub_pose, sub_img;
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

      const bool control = fresh && out.control;
      const bool fire = bc.algo.fire_enabled && control && out.fire;
      const double yaw = control ? out.yaw : pose.yaw;
      const double pitch = control ? out.pitch : pose.pitch;
      const std::string js = make_cmd_json(yaw, pitch, 0.0, fire, control);
      z_owned_bytes_t b;
      z_bytes_copy_from_buf(&b, reinterpret_cast<const uint8_t *>(js.data()), js.size());
      z_put(z_loan(s_img), z_loan(ke_cmd), z_move(b), NULL);

      if (bc.algo.visualize && !out.debug.empty()) {
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
        if (det_print++ % 5 == 0) {
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
      if (fire || out.n_det != 1 || k % 30 == 0) {
        std::printf(
            "[bridge] #%llu age=%.0fms det=%d sel=%d solved=%d q=%d st=%d dis=%.2f "
            "roll=%.3f w=%.3f lead=%.3f lat=%.1fms cmd=(%.2f,%.2f) err=%.4f ctrl=%d fire=%d "
            "nt=%d dlt=%.0f rc=%.2f\n",
            static_cast<unsigned long long>(k), age_ms, out.n_det, out.n_selected,
            out.solved ? 1 : 0, out.quality ? 1 : 0, out.status, out.buff_dis, out.roll,
            out.roll_velocity, out.lead_s, out.latency_ms, yaw * 180 / M_PI, pitch * 180 / M_PI,
            out.fire_err, control ? 1 : 0, fire ? 1 : 0, out.dual_targets, out.dual_delta_deg,
            out.rc_dist_m);
        std::fflush(stdout);
      }
      // 大符专项诊断（限流）：用来定位"黄/品红为什么还在换"。
      //   droll = 本帧两片 roll 与上一帧两片 roll 做【不依赖身份】最优配对后的两个残差。
      //           某一对接近 180 度 => 那片发生了 IPPE 双解翻转（相位翻到对面）。
      //   pick  = 两片各自选了第几个 IPPE 解；若某片的 pick 在 0/1 间来回跳，就是它。
      if (out.dual_targets == 2 && k % 10 == 0) {
        std::printf(
            "[dual] droll=(%.1f,%.1f)deg pick=(%d,%d) dres=%.1fdeg chord=%.3fm "
            "rcd(旧口径)=%.2fm dlt=%.0f\n",
            out.droll_deg[0], out.droll_deg[1], out.sol_pick_t[0], out.sol_pick_t[1],
            out.dual_resid_deg, out.chord_m, out.dual_rcd_m, out.dual_delta_deg);
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
