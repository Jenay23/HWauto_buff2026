#include "local_tracker.hpp"

#include <Eigen/Dense>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;

// 模拟器规则（power_rune.rs: VariableRotation::random）给出的硬边界：
//     speed(t) = a*sin(omega*t + phi) + (2.090 - a)
//     a     ∈ [0.780, 1.045]
//     omega ∈ [1.884, 2.000]
// 【平均角速度 = 2.090 - a 不是自由参数】—— 这正是之前"绿色乱飞"的根源：
// 旧模型里 c1*t 让斜率自由，噪声的斜率被吸进去；一旦冻结参数就变成随时间线性
// 增长的漂移，而且永不修正。
constexpr double kALo = 0.780, kAHi = 1.045;
constexpr double kWLo = 1.884, kWHi = 2.000;
constexpr double kMeanSpeed = 2.090;

constexpr int kASteps = 24;  // a 的网格点数
constexpr int kWSteps = 10;  // omega 的网格点数

constexpr double kMaxJumpDeg = 12.0;   // 单帧相位跳变上限 -> 清窗口
constexpr double kLockRmsDeg = 2.0;    // 形状残差小于它就锁形状
constexpr double kRelockRmsDeg = 4.0;  // 已锁后残差超过它就重新搜形状

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
  t_ref_ = 0.0;
}

// ---------------------------------------------------------------------------
// 相位模型（用上模拟器的参数约束）：
//     speed(t) = a*sin(w*t + phi) + (2.090 - a)
// 积分一次（tau = t - t_ref）：
//     phase(tau) = theta0 + (2.090 - a)*tau
//                  + p*(1 - cos(w*tau)) + q*sin(w*tau)
//   其中 p = (a/w)*cos(phi), q = (a/w)*sin(phi)
//
// 关键性质：对【固定的 (a, w)】只有 3 个线性未知数 (theta0, p, q) -> 闭式最小二乘；
// 而且平均角速度被 a 钉死，不再有自由漂移项。
// 所以把 (a, w) 当"形状"：拟合好就锁住；而 (theta0, p, q) 每帧用窗口内【全部点】重解
// —— 既有锚（不会漂），又平滑（线性部分是被平均过的）。
// ---------------------------------------------------------------------------
bool LocalRuneTracker::solveLinear(double a, double w, Eigen::Vector3d *coef, double *rms_rad) {
  const int n = static_cast<int>(h_.size());
  if (n < 4) return false;
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (const auto &s : h_) {
    const double tau = s.t - t_ref_;
    const double y = s.roll - (kMeanSpeed - a) * tau;  // 已知线性项移到右边
    Eigen::Vector3d phi(1.0, 1.0 - std::cos(w * tau), std::sin(w * tau));
    A += phi * phi.transpose();
    b += phi * y;
  }
  const Eigen::Vector3d c = A.ldlt().solve(b);
  if (!c.allFinite()) return false;
  double se = 0.0;
  for (const auto &s : h_) {
    const double tau = s.t - t_ref_;
    const double pred = (kMeanSpeed - a) * tau + c(0) + c(1) * (1.0 - std::cos(w * tau)) +
                        c(2) * std::sin(w * tau);
    const double e = s.roll - pred;
    se += e * e;
  }
  *coef = c;
  *rms_rad = std::sqrt(se / static_cast<double>(n));
  return true;
}

void LocalRuneTracker::refitShape() {
  double best_rms = 1e18;
  double best_a = 0.0, best_w = 0.0;
  Eigen::Vector3d best_c = Eigen::Vector3d::Zero();
  bool found = false;

  for (int ia = 0; ia <= kASteps; ++ia) {
    const double a = kALo + (kAHi - kALo) * ia / kASteps;
    for (int iw = 0; iw <= kWSteps; ++iw) {
      const double w = kWLo + (kWHi - kWLo) * iw / kWSteps;
      Eigen::Vector3d c;
      double rms = 0.0;
      if (!solveLinear(a, w, &c, &rms)) continue;
      if (!found || rms < best_rms) {
        found = true;
        best_rms = rms;
        best_a = a;
        best_w = w;
        best_c = c;
      }
    }
  }
  if (!found) return;
  lock_a_ = best_a;
  lock_w_ = best_w;
  lin_c_ = best_c;
  locked_ = 1;
  omega_ = best_w;
  amp_ = best_a;
  rms_deg_ = best_rms * 180.0 / kPi;
}

void LocalRuneTracker::push(double t, double roll) {
  // 跳变卫兵：单帧变化过大 -> 这一帧不可信 -> 清空窗口重新开始
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

  if (static_cast<int>(h_.size()) < 4) return;

  // 以窗口中点为时间原点（短窗口条件数更好）。t_ref_ 变了没关系：线性部分会重解，
  // 形状 (lock_a_/lock_w_) 不受影响。
  t_ref_ = 0.5 * (h_.front().t + h_.back().t);

  if (locked_ == 0) {
    refitShape();
  } else {
    // 已锁形状：每帧只重解 (theta0, p, q) —— 这就是"锚"
    Eigen::Vector3d c;
    double rms = 0.0;
    if (solveLinear(lock_a_, lock_w_, &c, &rms)) {
      lin_c_ = c;
      rms_deg_ = rms * 180.0 / kPi;
      if (rms_deg_ > kRelockRmsDeg) {  // 形状不再适用（换了一次激活）-> 重搜
        locked_ = 0;
        refitShape();
      }
    }
  }

  if (locked_ != 0) {
    const double tau = t - t_ref_;
    const double p = lin_c_(1), q = lin_c_(2);
    v_now_ =
        (kMeanSpeed - lock_a_) + lock_w_ * (p * std::sin(lock_w_ * tau) + q * std::cos(lock_w_ * tau));
    a_now_ = lock_w_ * lock_w_ * (p * std::cos(lock_w_ * tau) - q * std::sin(lock_w_ * tau));
  }
}

bool LocalRuneTracker::predict(double t_future, double *roll_out) const {
  if (locked_ == 0) return false;
  if (static_cast<int>(h_.size()) < min_n_) return false;
  const double tau = t_future - t_ref_;
  if (std::abs(tau) > 3.0) return false;  // 外推太远不信
  const double p = lin_c_(1), q = lin_c_(2);
  *roll_out = wrapPi((kMeanSpeed - lock_a_) * tau + lin_c_(0) +
                     p * (1.0 - std::cos(lock_w_ * tau)) + q * std::sin(lock_w_ * tau));
  return true;
}
