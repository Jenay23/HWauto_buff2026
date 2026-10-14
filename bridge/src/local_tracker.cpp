#include "local_tracker.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaxJumpDeg = 12.0;   // 单帧相位跳变上限 -> 清窗口
constexpr double kALo = 0.780, kAHi = 1.045;
constexpr double kWLo = 1.884, kWHi = 2.000;
constexpr double kMean = 2.090;
constexpr double kMaxSpeed = 2.090;
constexpr double I_FIT_S_MIN = 1.0;   // mode6: 攒够 1 s 才做一次网格搜索     // 绝对约束: v_max = 2.090 rad/s（与 a 无关）
constexpr double kMeanSpeedLo = 1.045;  // = 2.090 - 1.045
constexpr double kMeanSpeedHi = 1.310;  // = 2.090 - 0.780        // 官方: speed = a*sin(w t+phi) + (2.090-a)
constexpr int kASteps = 24;
constexpr int kWSteps = 10;
constexpr double kRidge = 1e-6;
constexpr double kLockRmsDeg = 2.0;
// （重锁门改成了成员 relock_rms_deg_，可由配置设置）

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
  // mode 6（滑行）：灭灯时保留拟合参数，只清样本 -> 之后用解析式继续外推。
  // 依据：模拟器里一组 (a, w) 一次激活只抽一次、之后不变（power_rune.rs），
  // 所以亮灯期拟合的模型在灭灯期依然有效 —— 这是物理延续，不是猜测。
  if (mode_ == 6 && locked_ != 0 && !lost_) {   // 未超时 -> 滑行；已 lost -> 落到下面完整重置
    if (!h_.empty()) last_data_t_ = h_.back().t;
    h_.clear();
    have_last_ = false;
    coasting_ = true;
    drop_run_ = 0;
    return;
  }
  h_.clear();
  have_last_ = false;
  v_now_ = a_now_ = omega_ = amp_ = 0.0;
  rms_deg_ = -1.0;
  span_deg_ = 0.0;
  drop_run_ = 0;
  locked_ = free_locked_ = 0;
  phys_valid_ = 0;
  sector_off_ = 0.0;
  ar_c_.clear();
  ekf_init_ = 0;
  ekf_rej_ = ekf_tot_ = ekf_miss_ = 0;
  lost_ = false;
  coasting_ = false;
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

// ======================= mode 7: physical model + robust LM =======================
//
// 模拟器 truth（power_rune.rs）：speed(t) = a*sin(w t) + (2.090 - a)，
// 积分得 phase(t) = th0 + sgn*[(2.090-a)t + (a/w)(cos(phi) - cos(w t + phi))]。
// t 的原点（激活时刻）未知，已经被 phi 吸收。参数: th0, a, w, phi，sgn 由数据趋势定。
//
// 为什么用 LM 而不是"网格 a,w + 线性解"（mode 0）：
//   - w 只有 1 维，且 [1.884,2.000] 很窄；LM 直接从初值收敛，不需要 24x10 网格。
//   - 残差用 Huber 权重，IPPE 翻转/换片造成的孤立大跳变自动降权，不需要丢帧。
//   - a,w 硬夹在官方区间内，模型永不漂出物理可行域。
double LocalRuneTracker::physEval(double t) const {
  const double c = (kMean - phys_a_) * t +
                   (phys_a_ / phys_w_) * (std::cos(phys_phi_) -
                                          std::cos(phys_w_ * t + phys_phi_));
  return phys_th0_ + phys_sgn_ * c;
}

void LocalRuneTracker::physFit() {
  const int n = static_cast<int>(h_.size());
  if (n < 6) { phys_valid_ = 0; return; }
  const double dt_span = h_.back().t - h_.front().t;
  // ★ 物理 span 上限（防"自转"）：真实相位速率 |dθ/dt| ≤ kMaxSpeed = 2.090 rad/s，
  //   所以一个窗口内的总 span 不可能超过 kMaxSpeed*dt_span。朴素增量解缠遇到
  //   IPPE 翻转/换片（实测 2.4% 的帧跳 ≈180°，且方向交替）会把每帧的 ±π 累加成
  //   "多圈自转"（实测坏段 span 1353° vs 真实 ~300°）——模型拟合了这个自转，
  //   未来打击点就绕符心转圈。这里一旦超出硬上限，判定本窗不可信（phys_valid_=0）。
  {
    const double max_span = kMaxSpeed * std::max(0.05, dt_span) * 1.5 + 0.3;
    if (std::abs(h_.back().roll - h_.front().roll) > max_span) {
      phys_valid_ = 0;
      return;
    }
  }
  if (dt_span < 1e-3) return;
  const double coarse = (h_.back().roll - h_.front().roll) / dt_span;
  double sgn = (coarse < 0.0) ? -1.0 : 1.0;

  // ★ 滑动窗口 + 参数热启动（近似"滑动窗口 BA 的平滑性"，零额外算法）：
  //   上一帧已经锁定的 (th0,a,w,phi) 直接作为本帧 LM 初值，不再从粗网格重搜。
  //   好处：① 输出随新数据连续微调，不做分支跳跃；② 省掉 5*6*12 网格，快一个量级。
  //   只在上一帧有效且距本帧 <0.5s 时热启动；否则回退粗网格自举。
  double best = 1e18, bth0 = 0.0, ba = 0.9, bw = 1.94, bphi = 0.0;
  const bool warm = (phys_valid_ != 0) && last_data_t_ > 0.0 &&
                    (h_.back().t - last_data_t_) < 0.5;
  if (warm) {
    bth0 = phys_th0_; ba = phys_a_; bw = phys_w_; bphi = phys_phi_; sgn = phys_sgn_;
  }
  for (int ia = 0; !warm && ia <= 4; ++ia) {
    const double a = kALo + (kAHi - kALo) * ia / 4.0;
    for (int iw = 0; iw <= 5; ++iw) {
      const double w = kWLo + (kWHi - kWLo) * iw / 5.0;
      for (int ip = 0; ip < 12; ++ip) {
        const double phi = 2.0 * kPi * ip / 12.0;
        double my = 0.0, mp = 0.0;
        for (const auto &s : h_) {
          my += s.roll;
          mp += sgn * ((kMean - a) * s.t +
                       (a / w) * (std::cos(phi) - std::cos(w * s.t + phi)));
        }
        my /= n; mp /= n;
        const double th0 = my - mp;
        double se = 0.0;
        for (const auto &s : h_) {
          const double p = th0 + sgn * ((kMean - a) * s.t +
                                        (a / w) * (std::cos(phi) - std::cos(w * s.t + phi)));
          se += (s.roll - p) * (s.roll - p);
        }
        if (se < best) { best = se; bth0 = th0; ba = a; bw = w; bphi = phi; }
      }
    }
  }

  // --- 稳健 LM ---
  double th0 = bth0, a = ba, w = bw, phi = bphi;
  double lam = 1e-3;
  auto eval = [&](double t, double A, double W, double P, double T0) {
    return T0 + sgn * ((kMean - A) * t +
                       (A / W) * (std::cos(P) - std::cos(W * t + P)));
  };
  auto sse = [&](double A_, double W_, double P_, double T0_) {
    double se = 0.0;
    for (const auto &s : h_) {
      const double r = s.roll - eval(s.t, A_, W_, P_, T0_);
      se += r * r;
    }
    return se;
  };
  double cur = sse(a, w, phi, th0);
  for (int it = 0; it < 50; ++it) {
    // 残差 + Huber 权重（IRLS）
    std::vector<double> r(n, 0.0), wt(n, 1.0);
    for (int i = 0; i < n; ++i) r[i] = h_[i].roll - eval(h_[i].t, a, w, phi, th0);
    {
      std::vector<double> ar(n);
      for (int i = 0; i < n; ++i) ar[i] = std::abs(r[i]);
      std::sort(ar.begin(), ar.end());
      double sig = 1.4826 * ar[n / 2];
      if (sig < 1e-4) sig = 1e-4;
      for (int i = 0; i < n; ++i) {
        const double u = r[i] / (2.5 * sig);
        wt[i] = 1.0 / std::sqrt(1.0 + u * u);
      }
    }
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    for (int i = 0; i < n; ++i) {
      const double t = h_[i].t;
      const double cw = std::cos(w * t + phi), sw = std::sin(w * t + phi);
      const double cp = std::cos(phi), sp = std::sin(phi);
      Eigen::Vector4d J;
      J(0) = 1.0;                                           // d/dth0
      J(1) = sgn * (-t + (1.0 / w) * (cp - cw));            // d/da
      J(2) = sgn * a * (t * sw / w - (cp - cw) / (w * w));  // d/dw
      J(3) = sgn * (a / w) * (-sp + sw);                    // d/dphi
      const double ww = wt[i] * wt[i];
      A += ww * J * J.transpose();
      b += ww * J * r[i];
    }
    for (int i = 0; i < 4; ++i) A(i, i) += lam * A(i, i) + 1e-12;
    const Eigen::Vector4d d = A.ldlt().solve(b);
    if (!d.allFinite()) break;
    const double nth0 = th0 + d(0);
    const double na = std::min(kAHi, std::max(kALo, a + d(1)));
    const double nw = std::min(kWHi, std::max(kWLo, w + d(2)));
    const double nphi = phi + d(3);
    const double nsse = sse(na, nw, nphi, nth0);
    if (nsse < cur) {
      th0 = nth0; a = na; w = nw; phi = nphi; cur = nsse;
      lam = std::max(lam * 0.4, 1e-6);
    } else {
      lam = std::min(lam * 4.0, 1e3);   // 拒绝：只加阻尼，参数不动
    }
    if (d.norm() < 1e-9 || lam >= 1e3) break;
  }

  phys_th0_ = th0;
  phys_a_ = a;
  phys_w_ = w;
  phys_phi_ = phi;
  phys_sgn_ = sgn;
  phys_valid_ = 1;
  locked_ = 1;
  omega_ = w;
  amp_ = a;
  // 稳健 rms（Huber 加权）
  {
    double se = 0.0;
    for (const auto &s : h_) {
      const double r = s.roll - physEval(s.t);
      se += r * r;
    }
    rms_deg_ = std::sqrt(se / n) * 180.0 / kPi;
  }
  // ★ 坏窗自愈（bridge/badwin 实验）：拟合"发疯"时（om/A 撞边界，rms 实测 200°+）：
  //   ① 模型判无效 —— 关键是让【创新门失效】。否则这个烂模型会把干净的新样本
  //      全部拒绝（phys_rej_run_ 攒到 15 才 reset，实测要 ~90 帧），窗口永远恢复不了；
  //   ② 只保留最近 reboot_keep_ 个样本，下一帧就能从干净样本重新自举（实测 2~3 帧）。
  //   好窗 rms 只有 0.5~8°，所以 30° 这个门不会误伤。
  if (rms_deg_ > bad_rms_deg_) {
    ++phys_heal_;
    phys_valid_ = 0;
    locked_ = 0;
    while (static_cast<int>(h_.size()) > reboot_keep_) h_.pop_front();
    return;
  }
  // 瞬时速度/加速度
  phys_sgn_ = sgn;
  const double t = h_.back().t;
  v_now_ = sgn * ((kMean - a) + a * std::sin(w * t + phi));
  a_now_ = sgn * a * w * std::cos(w * t + phi);
}

// ======================= mode 1: AR(p) =======================
void LocalRuneTracker::fitAR() {
  const int p = ar_p_;
  const int n = static_cast<int>(h_.size());
  if (n < p + 3) return;
  std::vector<double> y;
  y.reserve(n);
  for (const auto &s : h_) y.push_back(s.roll);
  // 去均值：相位绝对值可能离 0 很远，减掉均值能显著改善条件数
  double mu = 0.0;
  for (double v : y) mu += v;
  mu /= n;
  Eigen::MatrixXd A(n - p, p);
  Eigen::VectorXd b(n - p);
  for (int k = p; k < n; ++k) {
    for (int i = 0; i < p; ++i) A(k - p, i) = y[k - i - 1] - mu;
    b(k - p) = y[k] - mu;
  }
  Eigen::MatrixXd AtA = A.transpose() * A;
  const double tr = AtA.trace() / static_cast<double>(p);
  const double ridge = 1e-6 * (tr > 0 ? tr : 1.0);   // 相对 ridge，脏数据下也能解
  for (int i = 0; i < p; ++i) AtA(i, i) += ridge;
  Eigen::VectorXd c = AtA.ldlt().solve(A.transpose() * b);
  if (!c.allFinite()) {
    // 拟合失败 -> 退化为"等速外推"：theta(k) = 2*theta(k-1) - theta(k-2)
    ar_c_.clear();
    ar_c_.push_back(2.0);
    ar_c_.push_back(-1.0);
    ar_p_ = 2;
    rms_deg_ = -1.0;
    locked_ = 1;
    return;
  }
  ar_c_.assign(c.data(), c.data() + p);
  double se = 0.0;
  for (int k = p; k < n; ++k) {
    double pr = 0.0;
    for (int i = 0; i < p; ++i) pr += c(i) * (y[k - i - 1] - mu);
    pr += mu;
    se += (y[k] - pr) * (y[k] - pr);
  }
  rms_deg_ = std::sqrt(se / std::max(1, n - p)) * 180.0 / kPi;
  locked_ = 1;
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
  // ★ 用户思路：如果连续 hit_s_ 秒都没有进入空窗（一直没有 det==0），
  //   说明【刚打中了】—— 而命中恰恰证明"刚拟合出来的那组参数是对的"。
  //   所以把它快照下来，之后不要从零重新拟合。
  no_dark_s_ = t - last_dark_t_;
  if (mode_ == 6 && no_dark_s_ >= hit_s_ && !hit_confirmed_) {
    hit_confirmed_ = true;
    if (locked_ != 0) {
      snap_a_ = lock_a_;
      snap_w_ = lock_w_;
      snap_c_ = lin_c_;
      snap_t_ = t_ref_;
      have_snapshot_ = true;
    }
  }
  // ★★ 扇区重参考（RP-26Rune resolve_track_after_phase_jump 的移植）★★
  //   换片/换组时观测相位整体平移 k·(2π/5)（5 个扇区）。与其丢帧/重置，不如给观测
  //   加一个常量偏移 sector_off_，把新扇区折回原参考系：相位连续、窗口不清空、
  //   跨"命中换组"也不断（这正是我们之前 reset 掉窗口的原因）。
  if (mode_ == 7) {
    if (phys_valid_ && static_cast<int>(h_.size()) >= 20 && last_data_t_ > 0.0) {
      const double pred = physEval(t);
      const double sector = 2.0 * kPi / 5.0;
      const double base_res = std::abs(wrapPi(roll - sector_off_ - pred));
      if (base_res > phys_gate_rad_) {
        double best_res = base_res;
        int best_k = 0;
        for (int k = -2; k <= 2; ++k) {
          if (k == 0) continue;
          const double r = std::abs(wrapPi(roll - (sector_off_ + k * sector) - pred));
          if (r < best_res) { best_res = r; best_k = k; }
        }
        if (best_k != 0 && best_res < phys_gate_rad_) {
          sector_off_ += best_k * sector;   // 折回原参考系（不丢帧）
          ++sector_switch_;
        }
      }
    }
    roll = wrapPi(roll - sector_off_);
  }
  // 跳变卫兵（改）：原来一遇跳变就 reset() 清空【整个窗口】，
  // 代价约 0.4~0.5 s（要重新攒够 min_n/span），而实测跳变每几帧就来一次 ->
  // 窗口永远攒不起来（"拟合慢/拟合不出来"的直接原因）。
  // 现在：跳变只丢【这一帧】（不更新 last_raw_/last_unwrapped_，窗口保留），
  // 代价约 1 帧 (≈42 ms)。只有【连续】丢很多帧（说明真换了一次激活/长时间灭灯）
  // 才清窗。
  if (have_last_) {
    // ★ 阈值必须随帧间隔缩放。桥每帧耗时可变（实测 p99 ≈ 0.13 s），2.09 rad/s 下
    //   单帧真实位移可达 15°+；固定 12° 会把正常快帧误判成跳变。更糟的是 drop 时
    //   不推进 last_raw_，一次误判会让后续每一帧都继续"跳" -> 级联丢帧 -> reset。
    //   （实测旧逻辑 drop=3487 / 9358 帧 = 37%，窗口永远攒不起来。）
    const double dt_j = h_.empty() ? meanDt() : std::max(1e-3, t - h_.back().t);
    const double lim_deg =
        std::max(kMaxJumpDeg,
                 std::min(60.0, kMaxSpeed * dt_j * 180.0 / kPi * 1.6));  // 上限 60°
    if (std::abs(wrapPi(roll - last_raw_)) * 180.0 / kPi > lim_deg) {
      ++jump_dropped_;
      // ★ 仍然按解缠增量推进参考，避免"一次 drop -> 后续全部误判"的级联。
      last_unwrapped_ = last_unwrapped_ + wrapPi(roll - last_raw_);
      last_raw_ = roll;
      if (++drop_run_ > 30) reset();   // 连续丢很多才认为换激活/长时间异常
      return;                          // 本帧不进窗口
    }
  }
  // ★★ 模型锚定创新门（mode 7）★★
  //   物理模型一旦可信，它比任何单帧测量都可靠：真实相位连续、速率 ≤2.09 rad/s。
  //   凡是与模型单步预测相差超过 phys_gate_rad_ 的测量，都判为异常（IPPE 双解翻转、
  //   圆心换支、相位源切换），【一律不入窗】。这样窗口里只剩干净样本 —— 对比实验里
  //   "只修跳变卫兵"会让窗口装满换支帧，rms 反而涨到 49~92°。
  if (mode_ == 7 && phys_valid_ && static_cast<int>(h_.size()) >= 20 &&
      last_data_t_ > 0.0) {
    const double dtp = t - last_data_t_;
    if (dtp > 0.0 && dtp < 0.5) {
      const double resid = wrapPi(roll - physEval(t));
      if (std::abs(resid) > phys_gate_rad_) {
        ++phys_rej_;
        ++phys_rej_run_;
        if (have_last_) {   // 推进参考，避免"一次拒绝 -> 后续全被误判"的级联
          last_unwrapped_ = last_unwrapped_ + wrapPi(roll - last_raw_);
          last_raw_ = roll;
        }
        if (phys_rej_run_ > 15) reset();   // 连续 ~0.5 s 不符 -> 认定模型锁错，重新自举
        return;                             // 本帧不进窗口
      }
      phys_rej_run_ = 0;
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

  if (mode_ == 0 || mode_ == 6) {         // csine / mode6(滑行)
    t_ref_ = 0.5 * (h_.front().t + h_.back().t);
    if (locked_ == 0) {
      // ★ 有命中快照时：先用它试一次（参数很可能就是它）——成功就直接锁定，
      //   不必再等 1 s、也不必再全范围搜索。失败则照旧走"攒 1 s + 网格搜索"。
      bool snap_ok = false;
      if (mode_ == 6 && have_snapshot_) {
        Eigen::Vector3d c;
        double r = 0;
        const double keep_ref = t_ref_;
        t_ref_ = snap_t_;                    // 快照的相位基准必须一起用，否则相位会错
        if (solveLinear(snap_a_, snap_w_, &c, &r) && r >= 0.0 && r * 180.0 / M_PI <= 4.0) {
          lock_a_ = snap_a_;
          lock_w_ = snap_w_;
          lin_c_ = c;
          locked_ = 1;
          rms_deg_ = r;
          omega_ = snap_w_;
          amp_ = snap_a_;
          lost_ = false;
          coasting_ = false;
          last_data_t_ = t;
          snap_ok = true;
        } else {
          t_ref_ = keep_ref;
        }
      }
      if (!snap_ok) {
        // mode 6：先攒够 fit_s_ 秒再做一次网格搜索（"推理用 1 s"）；期间不预测
        const double span_s = h_.back().t - h_.front().t;
        if (mode_ != 6 || span_s >= I_FIT_S_MIN) {
          refitShape();
        }
      }
    } else if (mode_ == 6) {
      // 锁定后每帧重解线性部分；每 5 帧做一次局部微调。
      // 为什么：锁定时的 (a,Omega) 可能不准（实测常被顶到网格下界 1.884/0.780），
      // 而一次激活内又不能重来 —— 所以必须允许它慢慢收敛。
      // 原则：只接受明显变好的结果，且永不回退到未锁定 —— 只会更准，不会乱跳。
      Eigen::Vector3d c;
      double r = 0;
      if (solveLinear(lock_a_, lock_w_, &c, &r)) {
        lin_c_ = c;
        rms_deg_ = r * 180.0 / M_PI;
        lost_ = false;
        coasting_ = false;
        last_data_t_ = t;
        if (++tune_tick_ >= 5) {
          tune_tick_ = 0;
          const double da = (kAHi - kALo) / kASteps;
          const double dw = (kWHi - kWLo) / kWSteps;
          for (int ia = -2; ia <= 2; ++ia) {
            for (int iw = -2; iw <= 2; ++iw) {
              if (ia == 0 && iw == 0) continue;
              const double a2 = std::clamp(lock_a_ + ia * da, kALo, kAHi);
              const double w2 = std::clamp(lock_w_ + iw * dw, kWLo, kWHi);
              Eigen::Vector3d c2;
              double r2 = 0;
              if (!solveLinear(a2, w2, &c2, &r2)) continue;
              if (r2 < r * 0.995) {
                r = r2;
                c = c2;
                lock_a_ = a2;
                lock_w_ = w2;
                omega_ = w2;
                amp_ = a2;
              }
            }
          }
          lin_c_ = c;
          rms_deg_ = r * 180.0 / M_PI;
        }
      }
    } else {
      Eigen::Vector3d c;
      double r = 0;
      if (solveLinear(lock_a_, lock_w_, &c, &r)) {
        lin_c_ = c;
        lost_ = false; coasting_ = false; last_data_t_ = t;
        rms_deg_ = r * 180.0 / kPi;
        if (rms_deg_ > relock_rms_deg_) { locked_ = 0; refitShape(); }
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
  } else if (mode_ == 7) {                 // 物理约束模型 + 稳健 LM
    last_data_t_ = t;
    physFit();
  } else if (mode_ == 1 || mode_ == 4) {   // AR / AR+包络
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
  if (mode_ == 7) {   // 物理约束模型：拟合成功 + 跨度够 + 残差小
    // ★ rms 门必须有：坏窗（解缠自转）的拟合 rms 实测可达 139~286°，若不查 rms，
    //   这种"自转解"会被当成可用 -> 未来打击点绕符心转圈（用户看到的紫点转满圈）。
    //   好窗 rms 只有 0.5~8°，用 15° 门可以干净分开。
    return phys_valid_ != 0 && static_cast<int>(h_.size()) >= 8 &&
           span_deg_ >= min_span_deg_ && rms_deg_ >= 0.0 && rms_deg_ <= max_rms_deg_;
  }
  if (mode_ == 0) {
    return locked_ != 0 && static_cast<int>(h_.size()) >= 4 && rms_deg_ >= 0.0 &&
           rms_deg_ <= max_rms_deg_;
  }
  if (mode_ == 1) {
    return static_cast<int>(ar_c_.size()) > 0 && static_cast<int>(h_.size()) >= min_n_ &&
           span_deg_ >= min_span_deg_ && rms_deg_ >= 0.0 && rms_deg_ <= max_rms_deg_;
  }
  if (mode_ == 6) {   // 滑行：拟合成功即可用（滑行期间也算），超时则不可用
    return locked_ != 0 && !lost_;
  }
  if (mode_ == 5) {   // AR 裸预测：真正无门 —— 有 2 个样本就可用
    return static_cast<int>(h_.size()) >= 2;
  }
  if (mode_ == 4) {   // AR + 物理包络：不用 rms 门（AR 拟合脏数据必然 rms 大），
                      // 安全性由 predict() 里的"速度不超过 2.090 rad/s"保证
    return static_cast<int>(ar_c_.size()) > 0 && static_cast<int>(h_.size()) >= min_n_ &&
           span_deg_ >= min_span_deg_;
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
  if (mode_ == 6) {   // 滑行：冻结参数继续外推；超过 coast_s_ 报消失
    if (locked_ == 0) return false;
    if (last_data_t_ > 0.0 && (t_future - last_data_t_) > coast_s_) {
      lost_ = true;
      return false;
    }
  }
  if (mode_ == 7) {
    if (last_data_t_ > 0.0 && (t_future - last_data_t_) > 1.0) return false;
    *roll_out = wrapPi(physEval(t_future));
    return true;
  }
  if (mode_ == 0 || mode_ == 6) {
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
  if (mode_ == 5) {   // AR 裸预测
    if (ar_c_.empty()) {   // 还没有系数 -> 等速外推（保证一定给得出结果）
      if (h_.size() < 2) return false;
      const auto &s1 = h_[h_.size() - 1];
      const auto &s0 = h_[h_.size() - 2];
      const double v = (s1.roll - s0.roll) / std::max(1e-6, s1.t - s0.t);
      *roll_out = wrapPi(s1.roll + v * (t_future - s1.t));
      return true;
    }
    // 以下为你指定参数范围隐含的唯一约束 = 你给的两条参数范围隐含的 |v| <= 2.090 rad/s
    const int p5 = static_cast<int>(ar_c_.size());
    if (p5 == 0 || static_cast<int>(h_.size()) < p5 + 1) return false;
    std::vector<double> y;
    for (const auto &s5 : h_) y.push_back(s5.roll);
    int steps5 = static_cast<int>(std::lround((t_future - h_.back().t) / dt));
    if (steps5 < 0 || steps5 > 60) return false;
    const double max_step5 = kMaxSpeed * dt;   // 2.090 rad/s * dt
    for (int k = 0; k < steps5; ++k) {
      double pr = 0.0;
      for (int i = 0; i < p5; ++i) pr += ar_c_[i] * y[y.size() - i - 1];
      const double prev = y.back();
      double d = pr - prev;
      if (d > max_step5) d = max_step5;        // ← 只这一条，别的都不要
      if (d < -max_step5) d = -max_step5;
      y.push_back(prev + d);
    }
    *roll_out = wrapPi(y.back());
    return true;
  }
  if (mode_ == 4) {   // AR + 物理包络
    const int p = static_cast<int>(ar_c_.size());
    if (p == 0 || static_cast<int>(h_.size()) < p + 1) return false;
    std::vector<double> y;
    for (const auto &s : h_) y.push_back(s.roll);
    int steps = static_cast<int>(std::lround((t_future - h_.back().t) / dt));
    if (steps < 0 || steps > 60) return false;
    const double max_step = kMaxSpeed * dt;      // ★ 绝对约束：单步最多走这么多
    double sum_v = 0.0;
    for (int k = 0; k < steps; ++k) {
      double pr = 0.0;
      for (int i = 0; i < p; ++i) pr += ar_c_[i] * y[y.size() - i - 1];
      const double prev = y.back();
      double d = pr - prev;
      if (d > max_step) d = max_step;            // 夹住：瞬时角速度 <= 2.090 rad/s
      if (d < -max_step) d = -max_step;
      y.push_back(prev + d);
      sum_v += d / dt;
    }
    // 平均角速度应落在 [2.090-1.045, 2.090-0.780]；越界说明 AR 在瞎走 -> 退回等速外推
    const double mean_v = (steps > 0) ? sum_v / steps : 0.0;
    if (steps > 0 && (mean_v < kMeanSpeedLo || mean_v > kMeanSpeedHi)) {
      const double v = (mean_v > 0 ? kMeanSpeedHi : -kMeanSpeedHi) * (mean_v > 0 ? 1.0 : 1.0);
      y.back() = wrapPi(h_.back().roll + (mean_v > 0 ? kMeanSpeedHi : -kMeanSpeedHi) * (steps * dt));
      (void)v;
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
