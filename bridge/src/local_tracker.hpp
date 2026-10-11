#pragma once

#include <deque>

#include <Eigen/Core>

// ============================================================================
// LocalRuneTracker —— 桥自己的"局部相位 tracker"（大符专用，独立于 HW26 的 tracker）
// ============================================================================
//
// 为什么不直接用 HW26 的 tracker：
//   1. HW26 用 RANSAC 拟合 v(t) = A*sin(w*t+phi) + C，要求内点数 >= min_inliers。
//      它的每个假设只用 3 个点（最小样本）-> 3 个点的噪声直接进解 -> 需要很多很多
//      点才能凑够内点。实测一次点亮窗口约 58 帧，仍然 ready 不了。
//   2. 点亮窗口约 2.0 s < 正弦周期 2*pi/w ≈ 3.1~3.3 s，一次窗口只覆盖约 60% 个
//      周期 -> (A, w, phi, C) 这 4 个参数在【信息上】就定不准。换算法也救不了。
//
// 本模块只做【短时外推】真正需要的事：把相位看成光滑函数，用滑动窗口的二次最小
// 二乘同时估 (相位, 角速度, 角加速度)。只有 3 个未知数、窗口约 0.5 s ->
// 条件数良态 -> 6~8 个样本即可用（对应点亮后约 0.2~0.3 s）。
//
// 精度依据：模拟器的规则给出 |角加速度| <= a*Omega <= 1.045*2.0 = 2.09 rad/s^2。
// 二次模型外推 t 秒的截断误差约 0.5*|vdot|*t^2，外推 0.2 s 时约 0.04 rad (2.4 deg)。
//
// 注意：本模块只负责【相位】。转盘中心由调用方提供（桥已由两片位置联合解出）。
class LocalRuneTracker {
 public:
  void reset();

  // 喂一个观测。roll 是【观测系】里的相位（rad），内部自动解缠，可以跨 ±pi。
  void push(double t, double roll);

  // 能不能用：三个门同时满足
  //   1) 样本数 >= min_n（默认 12，约 0.4s）
  //   2) 累计相位跨度 >= min_span_deg（默认 30 度）—— 跨度不够时，正弦项与线性项
  //      几乎线性相关，二次拟合病态 -> 加速度被噪声放大 -> 外推乱跳
  //   3) 拟合 RMS 残差 <= max_rms_deg（默认 2.5 度）—— 数据不自洽时宁可不给预测
  // 实测教训：一开始只用 6 个样本就 ready，绿点很早就出现，但跟踪特别差。
  bool ready() const {
    // 已锁存正弦参数（残差够小过）+ 样本够 + 残差仍在门内
    return locked_ != 0 && static_cast<int>(h_.size()) >= 4 && rms_deg_ >= 0.0 &&
           rms_deg_ <= max_rms_deg_;
  }
  int n() const { return static_cast<int>(h_.size()); }

  // 预测 t_future 时刻的相位，结果 wrap 到 (-pi, pi]。未 ready 时返回 false。
  bool predict(double t_future, double *roll_out) const;

  // 最近一次拟合得到的角速度 / 角加速度（诊断用；未 ready 时为 0）
  double velocity() const { return v_now_; }
  double omega() const { return omega_; }
  double amp() const { return amp_; }
  bool locked() const { return locked_ != 0; }
  double accel() const { return a_now_; }

  void set_window(int n) { win_ = n; }
  void set_min_samples(int n) { min_n_ = n; }
  void set_min_span_deg(double d) { min_span_deg_ = d; }
  void set_max_rms_deg(double d) { max_rms_deg_ = d; }
  void set_ridge(double r) { ridge_ = r; }

  // 诊断
  double rms_deg() const { return rms_deg_; }
  double span_deg() const { return span_deg_; }

 private:
  struct Sample {
    double t;
    double roll;  // 解缠后的
  };

  // 约束正弦模型：对固定 (a, w) 只有 3 个线性未知数 (theta0, p, q) -> 闭式最小二乘
  bool solveLinear(double a, double w, Eigen::Vector3d *coef, double *rms_rad);
  // 网格搜 (a, w) 取残差最小者，并锁住形状
  void refitShape();

  std::deque<Sample> h_;
  double last_raw_ = 0.0;
  double last_unwrapped_ = 0.0;
  bool have_last_ = false;
  int win_ = 20;          // 约 0.6 s @30fps
  int min_n_ = 12;        // 约 0.4 s —— 实测 6 个太少，预测乱跳
  double min_span_deg_ = 30.0;
  double max_rms_deg_ = 2.5;
  // ridge：对 (速度, 加速度) 加小惩罚，数据撑不住时自动退化成"匀速外推"
  double ridge_ = 1e-3;
  double v_now_ = 0.0;
  double a_now_ = 0.0;
  // --- 形状锁存：(a, w) 锁住，(theta0,p,q) 每帧重解 ---
  int locked_ = 0;                                    // 形状是否已锁定
  double lock_a_ = 0.0;                               // 锁定的 a
  double lock_w_ = 0.0;                               // 锁定的 omega
  Eigen::Vector3d lin_c_ = Eigen::Vector3d::Zero();   // 每帧重解的 (theta0, p, q)
  double t_ref_ = 0.0;                                // 当前窗口的时间原点
  double sgn_ = 1.0;                                  // 转向符号（由窗口斜率定出）
  double omega_ = 0.0;                                // 诊断：最近拟合的 omega
  double amp_ = 0.0;                                  // 诊断：最近拟合的 a
  double rms_deg_ = -1.0;
  double span_deg_ = 0.0;
};
