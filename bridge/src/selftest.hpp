#pragma once

// 两个自检（README §3.3/§3.4 附近）：都不需要模拟器。
//   --selftest-center : 合成圆上位置序列，验 fit_fixed_radius_center（6/6 通过）
//   --selftest <png>  : 对一张静图只跑检测，验证模型泛化性

#include <string>

#include "config_io.hpp"

/// 圆拟合自检。返回 0 = 全部符合预期。
int run_selftest_center(const BridgeConfig &bc);

/// 单图检测自检。返回 0 = 有检出 / 4 = 无检出 / 3 = 读不到图 / 5 = 抛异常。
int run_detect_selftest(const BridgeConfig &bc, const std::string &png);
