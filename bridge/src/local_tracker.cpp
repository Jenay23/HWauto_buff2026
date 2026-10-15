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

double wrapPi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a <= -kPi) a += 2.0 * kPi;
  return a;
}
}  // namespace

void LocalRuneTracker::reset() {
  h_.clear();
  have_last_ = false;
  v_now_ = a_now_ = omega_ = amp_ = 0.0;
  rms_deg_ = -1.0;
  span_deg_ = 0.0;
  drop_run_ = 0;
  locked_ = 0;
  phys_valid_ = 0;
  sector_off_ = 0.0;
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
      // bridge/stuckfix: 必须就地清窗，不能只置 phys_valid_=0 然后 return。
      // 本函数在这里就 return，后面那套 rms>30 度的自愈根本走不到；而这种坏窗的 rms
      // 可以极小（实测 0.8 度），rms 门永远不触发 -> 坏样本永远留在窗口 ->
      // phys_valid_ 永远 0 -> push() 的模型创新门被禁用（要求 phys_valid_） ->
      // 窗口长不大、rdy 恒 0、rej/drop 冻结。实测卡死 22~55 条 [dual]（约 7~16 秒），
      // 症状 local n=12 rdy=0 rms=0.8deg span=267deg（12 样本 0.36s 却转 267 度）。
      // 短窗口 + 不可能的 span 本身就是确凿的坏样本，故不加 heal_min_n_ 下限
      //（函数入口已保证 n>=6）。只从头部弹到 span 恢复可能为止，最多弹到 reboot_keep_。
      ++phys_heal_;
      locked_ = 0;
      const int keep = std::max(4, reboot_keep_);
      while (static_cast<int>(h_.size()) > keep) {
        h_.pop_front();
        const double dtp = h_.back().t - h_.front().t;
        const double lim = kMaxSpeed * std::max(0.05, dtp) * 1.5 + 0.3;
        if (std::abs(h_.back().roll - h_.front().roll) <= lim) break;
      }
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
  //
  // ★★ 但必须加"样本数下限" heal_min_n_：实测自愈会在自举期【空转】——
  //   窗口刚重开时 n=9、span 只有 24~39°，4 参数物理模型本来就不可辨识（要接近一整圈），
  //   这种"短窗口拟合失败"是【正常自举】却被当成"模型发疯" -> 砍窗 -> 窗口更短 ->
  //   再失败 -> 再砍……实测 10 帧内连触发 4 次、整组 45 次，反而制造了新的坏窗。
  //   只在"窗口本来够长却还拟合失败"时才自愈。
  if (rms_deg_ > bad_rms_deg_ && static_cast<int>(h_.size()) >= heal_min_n_) {
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

// ======================= push / ready / predict =======================
void LocalRuneTracker::push(double t, double roll) {
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

  if (mode_ == 7) {                 // 物理约束模型 + 稳健 LM
    last_data_t_ = t;
    physFit();
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
  return false;
}

bool LocalRuneTracker::predict(double t_future, double *roll_out) const {
  if (!ready()) return false;
  const double dt = meanDt();
  if (mode_ == 7) {
    if (last_data_t_ > 0.0 && (t_future - last_data_t_) > 1.0) return false;
    *roll_out = wrapPi(physEval(t_future));
    return true;
  }
  return false;
}
