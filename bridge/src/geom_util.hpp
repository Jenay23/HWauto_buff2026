#pragma once

// 通用几何/设备助手 + [stage] 逐帧计时（P2-6：从 hw26_pipeline.cpp 原样搬出）。
// 这里放的是**没有算法语义**的东西：坐标系投影、角度归一、姿态转矩阵、OpenVINO 设备兜底。
// 相机挂装/坐标约定仍只有 hw26_pipeline.hpp 一个来源（gimbal_to_camera）。

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "hw26_pipeline.hpp"

namespace hw26_bridge {

// ---- [stage] 每帧耗时统计（letterbox / infer / decode / draw）----
struct StageAcc {
  static constexpr int kN = 4;
  const char *names[kN] = {"letterbox", "infer", "decode", "draw"};
  double acc[kN] = {0,0,0,0}; long cnt[kN] = {0,0,0,0}; long frames = 0;
  std::chrono::steady_clock::time_point t;
  void mark(int i) {
    auto now = std::chrono::steady_clock::now();
    if (i > 0) { acc[i-1] += std::chrono::duration<double,std::milli>(now-t).count(); ++cnt[i-1]; }
    t = now;
  }
  void report(double lat) {
    ++frames; if (frames % 100 != 0) return;
    double sum = 0;
    for (int i=0;i<kN;++i) sum += cnt[i] ? acc[i]/cnt[i] : 0.0;
    std::printf("[stage] 每帧平均(ms):");
    for (int i=0;i<kN;++i) std::printf(" %s=%.2f", names[i], cnt[i]?acc[i]/cnt[i]:0.0);
    std::printf(" 其余=%.2f 合计=%.2f (lat=%.2f)\n", lat-sum, sum, lat);
    std::fflush(stdout);
    for (int i=0;i<kN;++i) { acc[i]=0; cnt[i]=0; }
  }
};

/// 定义在 hw26_pipeline.cpp（唯一 owner）
extern StageAcc g_stage_acc;

// ---- 设备编译兜底：GPU/NPU 失败自动退回 CPU ----
// 设备编译的兜底：GPU/NPU 编译失败时自动退回 CPU，避免桥直接崩掉。
template <typename T>
inline ov::CompiledModel compile_with_fallback(ov::Core &core, const T &thing,
                                               const std::string &dev) {
  if (dev == "CPU") return core.compile_model(thing, dev);
  try {
    return core.compile_model(thing, dev);
  } catch (const std::exception &e) {
    std::cerr << "[hw26] 警告: 设备 " << dev << " 编译失败，自动回退 CPU: " << e.what() << std::endl;
    return core.compile_model(thing, "CPU");
  }
}

// ---- 投影助手（消除轴向歧义）----
// ---- 统一的投影助手（消除轴向歧义）----
// 云台系(X前 Y左 Z上) -> 相机系(OpenCV: X右 Y下 Z前)
// 委托给 hw26_pipeline.hpp 里的单一来源 gimbal_to_camera()，
// 不要再在这里手写公式——相机挂装一变这里就会静默错。
inline Eigen::Vector3f gimbal_to_cam(const Eigen::Vector3f &p_g) {
  return gimbal_to_camera(p_g);
}
// 相机系 -> 像素（模拟器畸变为 0）。depth = p_cam.z()
inline bool project_cam(const Eigen::Vector3f &p_cam, const cv::Mat &K, int *px, int *py) {
  if (p_cam.z() < 0.2f) return false;
  const double Z = static_cast<double>(p_cam.z());
  *px = static_cast<int>(std::lround(K.at<double>(0, 2) + K.at<double>(0, 0) * p_cam.x() / Z));
  *py = static_cast<int>(std::lround(K.at<double>(1, 2) + K.at<double>(1, 1) * p_cam.y() / Z));
  return true;
}

inline double wrap_pi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

inline Eigen::Matrix3f gimbal_rotation(float yaw, float pitch, float roll) {
  Eigen::Matrix3f R =
      (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
       Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()))
          .toRotationMatrix();
  return R;
}

}  // namespace hw26_bridge
