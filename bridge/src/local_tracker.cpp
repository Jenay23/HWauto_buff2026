#include "local_tracker.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxJumpDeg = 12.0;   // 单帧相位跳变上限 -> 清窗口
constexpr double kALo = 0.780, kAHi = 1.045;
constexpr double kWLo = 1.884, kWHi = 2.000;
constexpr double kMean = 2.090;        // 官方: speed = a*sin(w t+phi) + (2.090-a)
constexpr int kASteps = 24;
constexpr int kWSteps = 10;
constexpr double kRidge = 1e-6;
constexpr double kLockRmsDeg = 2.0;
constexpr double kRelockRmsDeg = 4.0;

double wrapPi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a <= -kPi) a += 2.0 * kPi;
  return a;
}
bool solve3(const Eigen::Matrix3d &A_in, const Eigen::Vector3d &b, Eigen::Vector3d *x) {
  Eigen::Matrix3d A = A_in;
  for (int i = 0; i < 3; ++i) A(i, i) += 1e-9;
  Eigen::Vector3d r = A.ldlt().solve(b);
  if (!r.allFinite()) return false;
  *x = r;
  return true;
}
}  // namespace

void LocalRuneTracker::reset() {
  h_.clear();
  have_last_ = false;
  v_now_ = a_now_ = omega_ = amp_ = 0.0;
  rms_deg_ = -1.0;
  span_deg_ = 0.0;
  drop_run_ = 0;
  locked_ = free_locked_ = 0;
  ar_c_.clear();
  ekf_init_ = 0;
  ekf_rej_ = ekf_tot_ = ekf_miss_ = 0;
  etau_ = 0.0;
}

void LocalRuneTracker::computeSpanAndDt() {
  if (h_.size() < 2) return;
  span_deg_ = std::abs(h_.back().roll - h_.front().roll) * 180.0 / kPi;
  const double dt = (h_.back().t - h_.front().t) / static_cast<double>(h_.size() - 1);
  if (dt > 1e-4 && dt < 0.5) dt_ema_ = dt;
}

double LocalRuneTracker::meanDt() const { return dt_ema_ > 1e-4 ? dt_ema_ : 1.0 / 30.0; }

// ======================= mode 0: csine（官方约束）=======================
bool LocalRuneTracker::solveLinear(double a, double w, Eigen::Vector3d *coef, double *rms_rad) {
  const int n = static_cast<int>(h_.size());
  if (n < 4) return false;
  const double coarse =
      (h_.back().roll - h_.front().roll) / (h_.back().t - h_.front().t + 1e-9);
  sgn_ = (coarse < 0.0) ? -1.0 : 1.0;
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (const auto &s : h_) {
    const double tau = s.t - t_ref_;
    const double y = s.roll - sgn_ * (kMean - a) * tau;
    Eigen::Vector3d phi(1.0, 1.0 - std::cos(w * tau), std::sin(w * tau));
    A += phi * phi.transpose();
    b += phi * y;
  }
  Eigen::Vector3d c;
  if (!solve3(A, b, &c)) return false;
  double se = 0.0;
  for (const auto &s : h_) {
    const double tau = s.t - t_ref_;
    const double p = sgn_ * (kMean - a) * tau + c(0) +
                     sgn_ * (c(1) * (1.0 - std::cos(w * tau)) + c(2) * std::sin(w * tau));
    se += (s.roll - p) * (s.roll - p);
  }
  *coef = c;
  *rms_rad = std::sqrt(se / static_cast<double>(n));
  return true;
}

void LocalRuneTracker::refitShape() {
  double best = 1e18, ba = 0, bw = 0;
  Eigen::Vector3d bc = Eigen::Vector3d::Zero();
  bool found = false;
  for (int ia = 0; ia <= kASteps; ++ia) {
    const double a = kALo + (kAHi - kALo) * ia / kASteps;
    for (int iw = 0; iw <= kWSteps; ++iw) {
      const double w = kWLo + (kWHi - kWLo) * iw / kWSteps;
      Eigen::Vector3d c;
      double r = 0;
      if (!solveLinear(a, w, &c, &r)) continue;
      if (!found || r < best) { found = true; best = r; ba = a; bw = w; bc = c; }
    }
  }
  if (!found) return;
  lock_a_ = ba; lock_w_ = bw; lin_c_ = bc; locked_ = 1;
  omega_ = bw; amp_ = ba; rms_deg_ = best * 180.0 / kPi;
}

// ======================= mode 3: free linear sine =======================
bool LocalRuneTracker::fitFree(Eigen::Vector4d *coef, double *w_out, double *t_ref_out) {
  const int n = static_cast<int>(h_.size());
  if (n < 4) return false;
  const double t_ref = 0.5 * (h_.front().t + h_.back().t);
  double best = 1e18;
  Eigen::Vector4d bc = Eigen::Vector4d::Zero();
  double bw = 0;
  bool found = false;
  for (int gi = 0; gi <= 64; ++gi) {
    const double w = kWLo + (kWHi - kWLo) * gi / 64;
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    for (const auto &s : h_) {
      const double tau = s.t - t_ref;
      Eigen::Vector4d phi(1.0, tau, std::sin(w * tau), std::cos(w * tau));
      A += phi * phi.transpose();
      b += phi * s.roll;
    }
    for (int i = 1; i < 4; ++i) A(i, i) += kRidge;
    const Eigen::Vector4d c = A.ldlt().solve(b);
    if (!c.allFinite()) continue;
    double se = 0.0;
    for (const auto &s : h_) {
      const double tau = s.t - t_ref;
      const double p = c(0) + c(1) * tau + c(2) * std::sin(w * tau) + c(3) * std::cos(w * tau);
      se += (s.roll - p) * (s.roll - p);
    }
    const double r = std::sqrt(se / n);
    if (!found || r < best) { found = true; best = r; bc = c; bw = w; }
  }
  if (!found) return false;
  *coef = bc; *w_out = bw; *t_ref_out = t_ref;
  omega_ = bw; amp_ = std::hypot(bc(2), bc(3)) * bw;
  rms_deg_ = best * 180.0 / kPi;
  return true;
}

// ======================= mode 1: AR(p) =======================
void LocalRuneTracker::fitAR() {
  const int p = ar_p_;
  const int n = static_cast<int>(h_.size());
  if (n < p + 3) return;
  std::vector<double> y;
  y.reserve(n);
  for (const auto &s : h_) y.push_back(s.roll);
  Eigen::MatrixXd A(n - p, p);
  Eigen::VectorXd b(n - p);
  for (int k = p; k < n; ++k) {
    for (int i = 0; i < p; ++i) A(k - p, i) = y[k - i - 1];
    b(k - p) = y[k];
  }
  Eigen::MatrixXd AtA = A.transpose() * A;
  for (int i = 0; i < p; ++i) AtA(i, i) += 1e-9;
  const Eigen::VectorXd c = AtA.ldlt().solve(A.transpose() * b);
  if (!c.allFinite()) return;
  ar_c_.assign(c.data(), c.data() + p);
  // 残差 -> rms_deg_
  double se = 0.0;
  for (int k = p; k < n; ++k) {
    double pr = 0.0;
    for (int i = 0; i < p; ++i) pr += c(i) * y[k - i - 1];
    se += (y[k] - pr) * (y[k] - pr);
  }
  rms_deg_ = std::sqrt(se / std::max(1, n - p)) * 180.0 / kPi;
  locked_ = 1;
  // 诊断用的速度/加速度
  const double dt = meanDt();
  if (n >= 3) {
    v_now_ = (y[n - 1] - y[n - 3]) / (2.0 * dt);
    a_now_ = (y[n - 1] - 2.0 * y[n - 2] + y[n - 3]) / (dt * dt);
  }
}

// ======================= mode 2: EKF =======================
bool LocalRuneTracker::ekfReady() const { return ekf_init_ != 0 && n() >= min_n_; }

void LocalRuneTracker::ekfStep(double z) {
  const double dt = meanDt();
  auto speed = [](const double *x, double tau) {
    return x[1] * std::sin(x[2] * tau + x[3]) + (kMean - x[1]);
  };
  if (!ekf_init_) {
    ex_[0] = z; ex_[1] = 0.9; ex_[2] = 1.94; ex_[3] = 0.0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) eP_[i][j] = 0.0;
    eP_[0][0] = 4e-3; eP_[1][1] = 2e-2; eP_[2][2] = 3e-3; eP_[3][3] = 3.0;
    ekf_init_ = 1;
    etau_ = 0.0;
    ekf_tot_++;
    return;
  }
  // 过程噪声
  const double qa = 1e-5, qw = 1e-6, qphi = 1e-5, qth = 5e-3;
  // 雅可比（只有第一行非零）
  const double s = std::sin(ex_[2] * etau_ + ex_[3]);
  const double cc = std::cos(ex_[2] * etau_ + ex_[3]);
  const double g[3] = {s - 1.0, ex_[1] * etau_ * cc, ex_[1] * cc};
  double F[4][4] = {{0}};
  for (int i = 0; i < 4; ++i) F[i][i] = 1.0;
  F[0][1] = g[0] * dt; F[0][2] = g[1] * dt; F[0][3] = g[2] * dt;
  double Q[4][4] = {{0}};
  Q[0][0] = qth * dt; Q[1][1] = qa; Q[2][2] = qw; Q[3][3] = qphi;
  // x' = x + f*dt
  const double sp = speed(ex_, etau_);
  ex_[0] += sp * dt;
  // P' = F P F^T + Q
  double FP[4][4] = {{0}}, Pn[4][4] = {{0}};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double v = 0.0;
      for (int k = 0; k < 4; ++k) v += F[i][k] * eP_[k][j];
      FP[i][j] = v;
    }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double v = 0.0;
      for (int k = 0; k < 4; ++k) v += FP[i][k] * F[j][k];
      Pn[i][j] = v + Q[i][j];
    }
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) eP_[i][j] = Pn[i][j];
  // 创新门（带地板：按测量噪声而不是状态不确定度）
  const double R = 1e-4, s_floor = 4e-3;
  const double S = eP_[0][0] + R + s_floor;
  const double nu = wrapPi(z - ex_[0]);
  ekf_tot_++;
  if (std::abs(nu) > 3.0 * std::sqrt(S)) {
    ekf_rej_++;
    ekf_miss_++;
    etau_ += dt;
    if (ekf_miss_ >= 12) {  // 连续拒太多 -> 只重设 theta
      ex_[0] = z;
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) eP_[i][j] = 0.0;
      eP_[0][0] = 4e-3; eP_[1][1] = 2e-2; eP_[2][2] = 3e-3; eP_[3][3] = 3.0;
      ekf_miss_ = 0;
    }
    return;
  }
  ekf_miss_ = 0;
  double K[4];
  for (int i = 0; i < 4; ++i) K[i] = eP_[i][0] / S;
  for (int i = 0; i < 4; ++i) ex_[i] += K[i] * nu;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) eP_[i][j] -= K[i] * eP_[0][j];
  ex_[1] = std::min(kAHi, std::max(kALo, ex_[1]));
  ex_[2] = std::min(kWHi, std::max(kWLo, ex_[2]));
  etau_ += dt;
  omega_ = ex_[2];
  amp_ = ex_[1];
  v_now_ = speed(ex_, etau_);
}

// ======================= push / ready / predict =======================
void LocalRuneTracker::push(double t, double roll) {
  // 跳变卫兵（改）：原来一遇跳变就 reset() 清空【整个窗口】，
  // 代价约 0.4~0.5 s（要重新攒够 min_n/span），而实测跳变每几帧就来一次 ->
  // 窗口永远攒不起来（"拟合慢/拟合不出来"的直接原因）。
  // 现在：跳变只丢【这一帧】（不更新 last_raw_/last_unwrapped_，窗口保留），
  // 代价约 1 帧 (≈42 ms)。只有【连续】丢很多帧（说明真换了一次激活/长时间灭灯）
  // 才清窗。
  if (have_last_) {
    const double jump = std::abs(wrapPi(roll - last_raw_)) * 180.0 / kPi;
    if (jump > kMaxJumpDeg) {
      ++jump_dropped_;
      if (++drop_run_ > 30) {  // 连续丢 >30 帧 -> 认为换激活/长时间无有效数据
        reset();
      }
      return;                  // ★ 不推入本帧，也不更新 last_*
    }
  }
  drop_run_ = 0;
  double unwrapped = roll;
  if (have_last_) unwrapped = last_unwrapped_ + wrapPi(roll - last_raw_);
  last_raw_ = roll;
  last_unwrapped_ = unwrapped;
  have_last_ = true;
  h_.push_back({t, unwrapped});
  while (static_cast<int>(h_.size()) > win_) h_.pop_front();
  computeSpanAndDt();
  if (static_cast<int>(h_.size()) < 4) return;

  if (mode_ == 0) {                       // csine
    t_ref_ = 0.5 * (h_.front().t + h_.back().t);
    if (locked_ == 0) {
      refitShape();
    } else {
      Eigen::Vector3d c;
      double r = 0;
      if (solveLinear(lock_a_, lock_w_, &c, &r)) {
        lin_c_ = c;
        rms_deg_ = r * 180.0 / kPi;
        if (rms_deg_ > kRelockRmsDeg) { locked_ = 0; refitShape(); }
      }
    }
    if (locked_) {
      const double tau = t - t_ref_;
      const double p = lin_c_(1), q = lin_c_(2);
      v_now_ = sgn_ * ((kMean - lock_a_) +
                       lock_w_ * (p * std::sin(lock_w_ * tau) + q * std::cos(lock_w_ * tau)));
      a_now_ = sgn_ * lock_w_ * lock_w_ *
               (p * std::cos(lock_w_ * tau) - q * std::sin(lock_w_ * tau));
    }
  } else if (mode_ == 1) {                 // AR
    fitAR();
  } else if (mode_ == 2) {                 // EKF
    ekfStep(unwrapped);
  } else {                                 // free linear sine
    Eigen::Vector4d c;
    double w = 0, tr = 0;
    if (fitFree(&c, &w, &tr)) {
      free_c_ = c; free_w_ = w; free_t_ref_ = tr; free_locked_ = 1;
      const double tau = t - tr;
      v_now_ = c(1) + w * (c(2) * std::cos(w * tau) - c(3) * std::sin(w * tau));
      a_now_ = -w * w * (c(2) * std::sin(w * tau) + c(3) * std::cos(w * tau));
    }
  }
}

bool LocalRuneTracker::ready() const {
  if (mode_ == 0) {
    return locked_ != 0 && static_cast<int>(h_.size()) >= 4 && rms_deg_ >= 0.0 &&
           rms_deg_ <= max_rms_deg_;
  }
  if (mode_ == 1) {
    return static_cast<int>(ar_c_.size()) > 0 && static_cast<int>(h_.size()) >= min_n_ &&
           span_deg_ >= min_span_deg_ && rms_deg_ >= 0.0 && rms_deg_ <= max_rms_deg_;
  }
  if (mode_ == 2) {
    return ekfReady();
  }
  return free_locked_ != 0 && static_cast<int>(h_.size()) >= min_n_ && rms_deg_ >= 0.0 &&
         rms_deg_ <= max_rms_deg_;
}

bool LocalRuneTracker::predict(double t_future, double *roll_out) const {
  if (!ready()) return false;
  const double dt = meanDt();
  if (mode_ == 0) {
    const double tau = t_future - t_ref_;
    if (std::abs(tau) > 3.0) return false;
    const double p = lin_c_(1), q = lin_c_(2);
    *roll_out = wrapPi(sgn_ * (kMean - lock_a_) * tau + lin_c_(0) +
                       sgn_ * (p * (1.0 - std::cos(lock_w_ * tau)) + q * std::sin(lock_w_ * tau)));
    return true;
  }
  if (mode_ == 1) {
    const int p = static_cast<int>(ar_c_.size());
    if (p == 0 || static_cast<int>(h_.size()) < p + 1) return false;
    std::vector<double> y;
    for (const auto &s : h_) y.push_back(s.roll);
    int steps = static_cast<int>(std::lround((t_future - h_.back().t) / dt));
    if (steps < 0 || steps > 60) return false;
    for (int k = 0; k < steps; ++k) {
      double pr = 0.0;
      for (int i = 0; i < p; ++i) pr += ar_c_[i] * y[y.size() - i - 1];
      y.push_back(pr);
    }
    *roll_out = wrapPi(y.back());
    return true;
  }
  if (mode_ == 2) {
    int steps = static_cast<int>(std::lround((t_future - h_.back().t) / dt));
    if (steps < 0 || steps > 60) return false;
    double x[4] = {ex_[0], ex_[1], ex_[2], ex_[3]};
    double tau = etau_;
    for (int k = 0; k < steps; ++k) {
      x[0] += (x[1] * std::sin(x[2] * tau + x[3]) + (kMean - x[1])) * dt;
      tau += dt;
    }
    *roll_out = wrapPi(x[0]);
    return true;
  }
  const double tau = t_future - free_t_ref_;
  if (std::abs(tau) > 3.0) return false;
  *roll_out = wrapPi(free_c_(0) + free_c_(1) * tau + free_c_(2) * std::sin(free_w_ * tau) +
                     free_c_(3) * std::cos(free_w_ * tau));
  return true;
}
