#include "hw26_pipeline.hpp"

#include <cmath>

namespace hw26_bridge {

// 从 hw26_pipeline.cpp 原样搬出（P2-3），逻辑一行未改。
// 声明仍在 hw26_pipeline.hpp，所以调用点不需要动。

bool solve_ballistic_no_drag(double d, double h, double v0, double g,
                             double *pitch_up_rad, double *fly_time_s) {
  if (!(d > 1e-3) || !(v0 > 1.0)) return false;
  const double v2 = v0 * v0;
  const double disc = v2 * v2 - g * (g * d * d + 2.0 * h * v2);
  if (disc < 0.0) return false;
  const double root = std::sqrt(disc);
  const double tan_theta = (v2 - root) / (g * d);  // 低伸解
  const double theta = std::atan(tan_theta);
  const double c = std::cos(theta);
  if (std::abs(c) < 1e-6) return false;
  if (pitch_up_rad) *pitch_up_rad = theta;
  if (fly_time_s) *fly_time_s = d / (v0 * c);
  return true;
}

}  // namespace hw26_bridge
