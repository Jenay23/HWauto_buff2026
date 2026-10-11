#include "local_tracker.hpp"

#include <Eigen/Dense>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxJumpDeg = 12.0;  // 单帧相位跳变上限（超过就丢样本+重置窗口）

// 模拟器规则给出的硬边界（与 HW26 的 big_buff_observer.ransac 完全一致）
constexpr double kOmegaLo = 1.884;
constexpr double kOmegaHi = 2.000;
constexpr int kOmegaSteps = 64;
constexpr double kRidge = 1e-6;
// 锁存门：残差小于它就认为这一版正弦可信，锁住参数
constexpr double kLockRmsDeg = 2.5;
// 解锁门：锁存后，若【一步预测】误差超过它，说明模型不再适用（换了一次激活）-> 重新锁
constexpr double kRelockErrDeg = 8.0;

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
  rms_deg_ = -1.0;
  span_deg_ = 0.0;
  omega_ = 0.0;
  amp_ = 0.0;
  locked_ = 0;
}

// ---------------------------------------------------------------------------
// 正弦模型拟合（与 HW26 同一族模型，但解法和它不同）
//   相位模型：  phase(t) = c0 + c1*t + c2*sin(w*t) + c3*cos(w*t)
//   它等价于 v(t) = a*sin(w*t+phi) + C 的积分形式。
//   对【固定 w】它对 (c0..c3) 线性 -> 闭式最小二乘；
//   再对 w 在 [1.884, 2.000]（模拟器/ HW26 给的硬范围）做网格搜索取残差最小者。
//
//   为什么比 HW26 的 RANSAC 好：
//     HW26 每个假设只用 3 个点（最小样本），3 个点的噪声直接进解，所以要很多点
//     才凑够内点；这里是【窗口内全部点】一起最小二乘，噪声被平均掉 sqrt(N)。
//   为什么锁存：参数在一次激活内固定（模拟器规则），拟合成功一次后没必要每帧
//     重搜；锁住即可，只有残差变大（换了一次激活）才该重拟合。
// ---------------------------------------------------------------------------
bool LocalRuneTracker::fit(Eigen::Vector4d *coef, double *omega_out, double *t_ref_out) {
  const int n = static_cast<int>(h_.size());
  if (n < 4) return false;

  // 以窗口【中点】为时间原点：短窗口下比用首/末点更能改善条件数
  const double t_ref = 0.5 * (h_.front().t + h_.back().t);

  double best_rms = 1e18;
  Eigen::Vector4d best_c = Eigen::Vector4d::Zero();
  double best_w = 0.0;
  bool found = false;

  for (int gi = 0; gi <= kOmegaSteps; ++gi) {
    const double w = kOmegaLo + (kOmegaHi - kOmegaLo) * gi / kOmegaSteps;
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    for (const auto &s : h_) {
      const double tau = s.t - t_ref;
      Eigen::Vector4d phi(1.0, tau, std::sin(w * tau), std::cos(w * tau));
      A += phi * phi.transpose();
      b += phi * s.roll;
    }
    A(1, 1) += kRidge;
    A(2, 2) += kRidge;
    A(3, 3) += kRidge;
    const Eigen::Vector4d c = A.ldlt().solve(b);
    if (!c.allFinite()) continue;

    double se = 0.0;
    for (const auto &s : h_) {
      const double tau = s.t - t_ref;
      const double pred =
          c(0) + c(1) * tau + c(2) * std::sin(w * tau) + c(3) * std::cos(w * tau);
      const double e = s.roll - pred;
      se += e * e;
    }
    const double rms = std::sqrt(se / static_cast<double>(n));
    if (!found || rms < best_rms) {
      found = true;
      best_rms = rms;
      best_c = c;
      best_w = w;
    }
  }
  if (!found) return false;

  *coef = best_c;
  *omega_out = best_w;
  *t_ref_out = t_ref;
  amp_ = std::hypot(best_c(2), best_c(3)) * best_w;  // 角速度振幅 a（诊断用）
  rms_deg_ = best_rms * 180.0 / kPi;
  return true;
}

void LocalRuneTracker::push(double t, double roll) {
  // 跳变卫兵：单帧变化过大 -> 这一帧不可信 -> 清空窗口重新开始
  // （依据：转盘最快 2.09 rad/s，30fps 下单帧最多 ~4 度）
  if (have_last_) {
    const double jump = std::abs(wrapPi(roll - last_raw_)) * 180.0 / kPi;
    if (jump > kMaxJumpDeg) reset();
  }
  double unwrapped = roll;
  if (have_last_) unwrapped = last_unwrapped_ + wrapPi(roll - last_raw_);
  last_raw_ = roll;
  last_unwrapped_ = unwrapped;
  have_last_ = true;

  h_.push_back({t, unwrapped});
  while (static_cast<int>(h_.size()) > win_) h_.pop_front();
  span_deg_ = std::abs(h_.back().roll - h_.front().roll) * 180.0 / kPi;

  Eigen::Vector4d c;
  double w = 0.0, t_ref = 0.0;
  if (fit(&c, &w, &t_ref)) {
    omega_ = w;
    if (locked_ == 0) {
      // 还没锁：残差够小就锁住这一版
      if (rms_deg_ >= 0.0 && rms_deg_ <= kLockRmsDeg) {
        locked_ = 1;
        lock_c_ = c;
        lock_w_ = w;
        lock_t_ref_ = t_ref;
      }
    } else {
      // 已锁：**参数保持冻结**（这是"锁存"的意义 —— 参数在一次激活内是固定的）。
      // 之前这里每帧都覆盖 lock_*，导致参数一直在变、预测跟着抖。
      // 只有当【一步预测】误差变大时，才认为换了一次激活，重新锁。
      const double tau = t - lock_t_ref_;
      const double pred = lock_c_(0) + lock_c_(1) * tau + lock_c_(2) * std::sin(lock_w_ * tau) +
                          lock_c_(3) * std::cos(lock_w_ * tau);
      const double err_deg = std::abs(wrapPi(unwrapped - pred)) * 180.0 / kPi;
      if (err_deg > kRelockErrDeg) {
        lock_c_ = c;
        lock_w_ = w;
        lock_t_ref_ = t_ref;
      }
    }
    // 诊断用的瞬时速度/加速度：锁存后按【锁存模型】算，避免诊断量自己抖
    const double tr = (locked_ != 0) ? lock_t_ref_ : t_ref;
    const Eigen::Vector4d &cc = (locked_ != 0) ? lock_c_ : c;
    const double ww = (locked_ != 0) ? lock_w_ : w;
    const double tau2 = t - tr;
    v_now_ = cc(1) + ww * (cc(2) * std::cos(ww * tau2) - cc(3) * std::sin(ww * tau2));
    a_now_ = -ww * ww * (cc(2) * std::sin(ww * tau2) + cc(3) * std::cos(ww * tau2));
  }
}

bool LocalRuneTracker::predict(double t_future, double *roll_out) const {
  if (locked_ == 0) return false;
  if (static_cast<int>(h_.size()) < min_n_) return false;
  const double w = lock_w_;
  const double tau = t_future - lock_t_ref_;
  if (std::abs(tau) > 3.0) return false;  // 外推太远不信
  *roll_out =
      wrapPi(lock_c_(0) + lock_c_(1) * tau + lock_c_(2) * std::sin(w * tau) +
             lock_c_(3) * std::cos(w * tau));
  return true;
}
