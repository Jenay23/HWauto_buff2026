#pragma once

#include <deque>

#include <Eigen/Core>

// ============================================================================
// LocalRuneTracker —— 桥自己的"局部相位 tracker"（大符专用）
// ============================================================================
//
// 【可选四种估计器】由配置 local_model 切换（改配置即可，不用重编译）：
//   0 = csine  ：官方约束模型 phase = theta0 + sgn*[(2.090-a)t + p(1-cos wt) + q sin wt]
//                (a,w) 在官方边界内【网格搜索】+ 每帧重解线性部分 (theta0,p,q)
//                离线实测（干净数据）227/341/455 ms 误差 2.30/3.69/5.09 度，最好
//   1 = AR(p)  ：纯数据驱动 theta(t)=sum c_i theta(t-i)，系数最小二乘 + 递推多步
//                离线实测干净数据 114 ms 0.5~0.75 度最好，长程变差
//   2 = EKF    ：状态 [theta,a,w,phi]，过程用官方速度公式，观测 theta，带创新门
//                注意：离线实测它【不好】—— 只测 theta 且窗口不足 1 周期时 a/w 不可辨识，
//                它和批拟合撞同一堵墙。留在这里是为了对比，不建议作为主方案。
//   3 = free   ：自由线性正弦 phase = c0 + c1*t + c2*sin(wt) + c3*cos(wt)（w 网格）
//                c1 自由 -> 能适配任意转向，但会吸收噪声斜率 -> 会漂
//
// 共同点：只负责【相位】；转盘中心由调用方提供。
// 为什么不用 HW26 的 tracker：见 README/提交历史（RANSAC 3 点最小样本 + 内点门，
// 实测一次点亮窗口约 58 帧仍 ready 不了）。
class LocalRuneTracker {
 public:
  void reset();

  // 喂一个观测。roll 是【观测系】里的相位（rad），内部自动解缠。
  void push(double t, double roll);

  // 能不能用（不同模式判据不同）
  bool ready() const;
  int n() const { return static_cast<int>(h_.size()); }

  // 预测 t_future 时刻的相位（wrap 到 (-pi,pi]）。未 ready 返回 false。
  bool predict(double t_future, double *roll_out) const;

  // 诊断
  double velocity() const { return v_now_; }
  double accel() const { return a_now_; }
  double omega() const { return omega_; }   // csine/free: 拟合的 w；EKF: 状态里的 w
  double amp() const { return amp_; }       // csine: 拟合的 a；EKF: 状态里的 a
  bool locked() const { return locked_ != 0; }
  double rms_deg() const { return rms_deg_; }
  double span_deg() const { return span_deg_; }
  int ekf_rej() const { return ekf_rej_; }
  int jump_dropped() const { return jump_dropped_; }
  bool lost() const { return lost_; }
  bool hit_confirmed() const { return hit_confirmed_; }   // 连续 >=hit_s_ 有靶标 -> 视为命中
  // 灭灯分两种：短暂缺失 -> 滑行（保留参数）；真正灭灯（超过 coast_s_）-> 清空参数。
  // 依据：模拟器每次【新激活】都会重抽 (a, w)，所以旧参数跨激活没有意义。
  void on_dark(double t) {
    if (mode_ == 6 && locked_ != 0 && last_data_t_ > 0.0 && (t - last_data_t_) > coast_s_) {
      reset();                 // 清空（含快照/命中标记）
      lost_ = true;            // 报靶标消失（reset 会清掉，所以再置一次）
      lost_reported_ = true;
      return;
    }
    last_dark_t_ = t;
    hit_confirmed_ = false;
    reset();                   // mode6 且未超时 -> reset 内部会走滑行分支（保留参数）
  }
  bool lost_reported() const { return lost_reported_; }
  bool coasting() const { return coasting_; }
  int ekf_tot() const { return ekf_tot_; }

  void set_window(int n) { win_ = n; }
  void set_min_samples(int n) { min_n_ = n; }
  void set_min_span_deg(double d) { min_span_deg_ = d; }
  void set_max_rms_deg(double d) { max_rms_deg_ = d; }
  void set_relock_rms_deg(double d) { relock_rms_deg_ = d; }
  void set_rms_gates(double d) { max_rms_deg_ = d; relock_rms_deg_ = d; }
  void set_coast_s(double s) { coast_s_ = s; }
  void set_ridge(double r) { ridge_ = r; }
  void set_mode(int m) {
    mode_ = m;
    // mode6 先攒 fit_s_=1.0 s 才做网格搜索 -> 窗口必须装得下 1.0 s，
    // 否则 span 永远到不了 1.0 s -> 永不锁定 -> ready() 永远 false。
    if (m == 6 && win_ < 48) win_ = 48;   // ≈1.6~2.0 s @30~24fps
  }
  int mode() const { return mode_; }

 private:
  struct Sample {
    double t;
    double roll;  // 解缠后的
  };

  // --- 通用 ---
  void computeSpanAndDt();
  double meanDt() const;

  // --- mode 0: csine ---
  bool solveLinear(double a, double w, Eigen::Vector3d *coef, double *rms_rad);
  void refitShape();

  // --- mode 3: free linear sine ---
  bool fitFree(Eigen::Vector4d *coef, double *w_out, double *t_ref_out);

  // --- mode 1: AR ---
  void fitAR();  // 更新 ar_c_

  // --- mode 2: EKF（手写 4x4）---
  void ekfStep(double z);
  bool ekfReady() const;

  std::deque<Sample> h_;
  double last_raw_ = 0.0;
  double last_unwrapped_ = 0.0;
  bool have_last_ = false;
  int mode_ = 0;
  int win_ = 20;          // 约 0.6 s @30fps
  int min_n_ = 12;
  double min_span_deg_ = 30.0;
  double max_rms_deg_ = 2.5;
  double relock_rms_deg_ = 4.0;
  int tune_tick_ = 0;   // 超过它就重新搜 (a,w)（原来写死 4.0）
  double ridge_ = 1e-3;
  double v_now_ = 0.0;
  double a_now_ = 0.0;
  double omega_ = 0.0;
  double amp_ = 0.0;
  double rms_deg_ = -1.0;
  double span_deg_ = 0.0;
  double dt_ema_ = 1.0 / 30.0;
  int jump_dropped_ = 0;
  bool coasting_ = false;
  mutable bool lost_ = false;
  bool lost_reported_ = false;   // mutable: predict() 是 const，超时要在里面置位
  double last_data_t_ = 0.0;
  double coast_s_ = 0.2;   // 只容忍【短暂缺失】；超过它 = 真灭灯 -> 清空参数
  double hit_s_ = 2.6;     // 连续这么久都没进空窗 -> 判定命中（激活没结束）
  bool hit_confirmed_ = false;
  double last_dark_t_ = 0.0;
  double no_dark_s_ = 0.0;
  bool have_snapshot_ = false;      // 命中确认时的参数快照（当 warm start 用）
  double snap_a_ = 0.0, snap_w_ = 0.0, snap_t_ = 0.0;
  Eigen::Vector3d snap_c_ = Eigen::Vector3d::Zero();   // 灭灯后继续外推的上限（= 亮灯 2.5s − 拟合 1s）
  double fit_s_ = 1.0;     // 先攒够 1.0 s 数据，再做一次（较贵的）网格搜索
  int drop_run_ = 0;       // 连续被丢的帧数

  // csine
  int locked_ = 0;
  double lock_a_ = 0.0;
  double lock_w_ = 0.0;
  Eigen::Vector3d lin_c_ = Eigen::Vector3d::Zero();
  double t_ref_ = 0.0;
  double sgn_ = 1.0;

  // free
  int free_locked_ = 0;
  Eigen::Vector4d free_c_ = Eigen::Vector4d::Zero();
  double free_w_ = 0.0;
  double free_t_ref_ = 0.0;

  // AR
  int ar_p_ = 4;
  std::deque<double> ar_c_;

  // EKF: x = [theta, a, w, phi]
  double ex_[4] = {0, 0.9, 1.94, 0.0};
  double eP_[4][4] = {{0}};
  double etau_ = 0.0;
  int ekf_init_ = 0;
  int ekf_rej_ = 0;
  int ekf_tot_ = 0;
  int ekf_miss_ = 0;
};
