#pragma once

// 桥的运行配置：算法参数（hw26_bridge::Config）+ Zenoh 话题名 + 图像过期门。
// 拆出这个模块是为了让 main.cpp 只剩"线程/主循环"，见 README §8。

#include <string>

#include "hw26_pipeline.hpp"

struct BridgeConfig {
  hw26_bridge::Config algo;
  std::string image_topic = "rt/image_raw";
  std::string pose_topic = "rt/gimbal_pose";
  std::string cmd_topic = "cmd/gimbal";
  std::string rune_kpts_topic = "rt/rune_kpts";   // 模拟器真值 8 点（只读订阅）
  std::string rune_stat_topic = "rt/stat";        // 模拟器命中统计 {launch,accurate,pct}（只读）
  double max_image_age_ms = 300.0;  // 超过则不发控制（保持云台）
};

/// 读扁平 yaml（`key: value` / `key: [a, b, c]` / `# 注释`）并填好默认值。
/// 抛 std::runtime_error（打不开文件）或 std::invalid_argument（键名不认识）。
BridgeConfig load_config(const std::string &path);
