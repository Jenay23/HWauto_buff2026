#pragma once

// 视频回放（--replay）：不需要模拟器，用来复现/验证算法链。
// 它同时是重构的守门员：同一段视频跑两遍，除 latency_ms 外 CSV 应逐列一致。

#include <string>

#include "config_io.hpp"

int run_replay(const BridgeConfig &bc, const std::string &video_path, int max_frames, bool headless,
               bool closed_loop);
