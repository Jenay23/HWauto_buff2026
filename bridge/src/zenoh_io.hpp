#pragma once

// Zenoh 收发（Reborn/旧模拟器这条通道）：
//   rt/image_raw  -> BIMG 帧 -> 最新帧邮箱（覆盖式，不排队）-> 主循环
//   rt/gimbal_pose-> JSON -> 姿态邮箱
//   rt/rune_kpts  -> JSON 真值 8 点（只读，写日志）
//   rt/stat       -> JSON 命中统计（只读，写 *.rtstat.csv）
//   cmd/gimbal    <- JSON 指令（make_cmd_json）
//
// 这里只做"字节<->结构体"，不含任何算法。抽取动机见 README §8 的模块清单。

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <zenoh.h>

#include "app_state.hpp"   // LatestFrame::take() 里要看 g_stop
#include "config_io.hpp"

struct GimbalPose {
  float yaw = 0, pitch = 0, roll = 0;
};

struct BimgFrame {
  int64_t timestamp_ns = 0;
  uint32_t w = 0, h = 0;
  const uint8_t *rgb = nullptr;
  size_t rgb_len = 0;
};

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

// ---- 共享状态（主循环与回调两头都在用）----
extern LatestFrame g_frame;                  // 图像最新帧邮箱
extern std::mutex g_pose_mu;                 // 姿态
extern GimbalPose g_pose;
extern std::atomic<uint64_t> g_pose_count;   // 收到的姿态条数（打印汇总用）
extern std::atomic<uint64_t> g_img_count;    // 收到的图像帧数

extern std::mutex g_kpts_mu;                 // rt/rune_kpts 真值（只留最新一帧写日志）
extern std::string g_kpts_json;
extern std::ofstream g_kpts_log;
extern std::mutex g_stat_mu;                 // rt/stat 命中统计
extern std::string g_stat_path;
extern long long g_stat_launch, g_stat_hits;

// ---- 解析 / 拼装 ----
bool parse_pose(const uint8_t *d, size_t n, GimbalPose *out);
bool parse_bimg(const uint8_t *d, size_t n, BimgFrame *f);
std::string make_cmd_json(double yaw, double pitch, double roll, bool fire, bool control);

// ---- Zenoh 订阅回调（由 main 在声明订阅者时传进去）----
void on_pose(struct z_loaned_sample_t *sample, void *);
void on_image(struct z_loaned_sample_t *sample, void *);
void on_rune_kpts(struct z_loaned_sample_t *sample, void *);
void on_rune_stat(struct z_loaned_sample_t *sample, void *);

/// --capture：订阅图像话题，等 on_image 落盘一张 PNG 后返回。
int run_capture(const BridgeConfig &bc, const std::string &out_path, int timeout_s);
