#pragma once

#include <cmath>

/// 把角度归一到 (-π, π]。原先只存在于 main.cpp，replay 也要用，所以提出来共享。
inline double wrap_pi_local(double x) {
  while (x > M_PI) x -= 2 * M_PI;
  while (x < -M_PI) x += 2 * M_PI;
  return x;
}
