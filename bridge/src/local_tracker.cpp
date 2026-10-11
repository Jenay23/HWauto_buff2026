#include "local_tracker.hpp"

#include <Eigen/Dense>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;

double wrapPi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a <= -kPi) a += 2.0 * kPi;
  return a;
}
}  // namespace

void LocalRuneTracker::reset() {
  h_.clear();
  have_last_ = false;
  v_now_ = 0.0;
  a_now_ = 0.0;
}

void LocalRuneTracker::push(double t, double roll) {
  double unwrapped = roll;
  if (have_last_) {
    // 解缠：把与上一个样本的差映射到 (-pi, pi]，累加到连续相位上
    unwrapped = last_unwrapped_ + wrapPi(roll - last_raw_);
  }
  last_raw_ = roll;
  last_unwrapped_ = unwrapped;
  have_last_ = true;

  h_.push_back({t, unwrapped});
  while (static_cast<int>(h_.size()) > win_) h_.pop_front();

  Eigen::Vector3d c;
  double t_ref = 0.0;
  if (fit(&c, &t_ref)) {
    v_now_ = c(1);        // 相位对时间的一阶导（局部时刻）
    a_now_ = 2.0 * c(2);
    // 诊断：拟合残差 RMS 与窗口内的相位跨度
    double se = 0.0;
    for (const auto &s : h_) {
      const double tau = s.t - t_ref;
      const double pred = c(0) + c(1) * tau + c(2) * tau * tau;
      const double e = s.roll - pred;
      se += e * e;
    }
    rms_deg_ = std::sqrt(se / static_cast<double>(h_.size())) * 180.0 / kPi;
    // 取绝对值：转盘可能正转也可能反转。之前用带符号的值，导致反转时
    // span 恒为负，span>=min_span 的判据永远不成立 -> 永不 ready（实测确认）。
    span_deg_ = std::abs(h_.back().roll - h_.front().roll) * 180.0 / kPi;
  }
  if (h_.size() < 3) {
    rms_deg_ = -1.0;
    span_deg_ = 0.0;
  }
}

bool LocalRuneTracker::fit(Eigen::Vector3d *coef, double *t_ref) const {
  if (static_cast<int>(h_.size()) < 3) return false;

  // 以最后一个样本为时间原点 —— 短窗口下这是改善条件数的关键
  const double tr = h_.back().t;
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (const auto &s : h_) {
    const double tau = s.t - tr;
    const Eigen::Vector3d phi(1.0, tau, tau * tau);
    A += phi * phi.transpose();
    b += phi * s.roll;
  }
  // 3x3 对称正定，用 LDLT
  // ridge：只惩罚 (速度, 加速度) 两项，不惩罚常数项（相位本身是无界的）
  A(1, 1) += ridge_;
  A(2, 2) += ridge_;
  const Eigen::Vector3d c = A.ldlt().solve(b);
  if (!c.allFinite()) return false;
  *coef = c;
  *t_ref = tr;
  return true;
}

bool LocalRuneTracker::predict(double t_future, double *roll_out) const {
  if (!ready()) return false;
  Eigen::Vector3d c;
  double t_ref = 0.0;
  if (!fit(&c, &t_ref)) return false;
  const double dt = t_future - t_ref;
  *roll_out = wrapPi(c(0) + c(1) * dt + c(2) * dt * dt);
  return true;
}
