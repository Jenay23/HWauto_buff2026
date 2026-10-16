#include "replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "app_state.hpp"
#include "hw26_pipeline.hpp"
#include "math_util.hpp"

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
    // 圆心来源诊断（center_source=1 时才有意义）：n = 参与拟合的样本数，rms = 拟合残差
    if (bc.algo.center_source == 1 && n % 10 == 0) {
      std::printf("[ctr] src=%d n=%d rms=%.4fm\n", out.center_src, out.center_n,
                  out.center_rms_m);
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
    std::printf("\n[replay] === 指令增量自相关（极限环判据，对齐的分析口径）===\n"
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
