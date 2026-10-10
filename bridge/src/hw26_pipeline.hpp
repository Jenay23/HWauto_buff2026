// hw26_pipeline.hpp — 把 HWauto_buff2026 的算法链接到 Daedalus 模拟器上
//
// 设计：
//   * 只替换 HW26 的 BuffDetector（TensorRT 类），改用 OpenVINO 跑同一个 buff.onnx；
//     解码仍用 HW26 自己的 letterbox_image() + decode_detections()。
//   * Selector / PnPSolver / Tracker / RANSAC / Predictor 全部原样复用 HW26 的类。
//   * HW26 缺的"后半截"在本文件补齐：观测系变换 → 弹道 → 云台角 → 开火判定。
//
// 坐标系约定（与模拟器 POWER_RUNE_INTERFACE.md 一致）：
//   外部系 X 前 / Y 左 / Z 上；`pitch < 0` 表示抬枪。
//   rt/gimbal_pose 的 yaw/pitch/roll 与 cmd/gimbal 的 yaw/pitch 同约定。
//   cmd.pitch = -(弹道仰角 + pitch_offset_rad)，pitch_offset 默认 -20°，
//   用于抵消枪管相对云台固定上仰 20°（与 ④ sp25_buff 桥同一条符号链，已在模拟器上验证过）。
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include "buff_algo/detector/detector_common.hpp"
#include "buff_algo/locate/buff_pnp_solver.hpp"
#include "buff_algo/predictor/buff_predictor.hpp"
#include "buff_algo/selector/buff_selector.hpp"
#include "buff_algo/tracker/buff_tracker.hpp"

namespace hw26_bridge {

// 观测向量送入 Tracker 前使用的参考系
enum class RefFrame {
  kFace,     // ★默认★ 符面系：x = 符面法向(车体系里静止)，y-z 为面内基。
             //   同时满足两件事：(1) e=R*(0,0,-0.7) 落在面内 -> e.x=0 精确成立，
             //   HW26 的 roll/predict_position 完全精确；(2) 坐标系不随云台转
             //   -> 不存在"观测挂在自身输出上"的正反馈，不会自激抖动。
  kChassis,  // 车体系：静止，但符在侧方时 e.x != 0 -> ②的固定平面重建有 16~23cm 偏差
  kGimbal,   // 相机/云台系：**会自激**。相机与枪管固定，云台系=瞄准系，
             //   把观测放在这里等于把观测挂在自己的输出上 -> 正反馈抖动。
             //   仅保留用于对照复现该现象。
};

// 检测器后端
enum class DetectorKind {
  kSp25,  // sp_vision_25 的 yolo11_buff_int8（④ 已在模拟器上验证 solved≈94%）
  kHw26,  // HWauto 自带的 YOLO11s-pose（实测在模拟器上输出满画面乱飘的误检，不可用）
};

// 单个检出的类别摘要（用于诊断"检出但被过滤"）
struct DetInfo {
  int color = -1;   // 0=BLUE 1=RED
  int label = -1;   // 0=未激活 1=已激活
  float confidence = 0.0f;
  float cx = 0.0f, cy = 0.0f;
};

struct Config {
  DetectorKind detector = DetectorKind::kSp25;
  // 模型与内参
  std::string model_path;       // HW26 模型（detector=kHw26 时用）
  std::string model_path_sp25;  // sp25 模型（detector=kSp25 时用）
  // sp25 的物体点半边长（相对扇叶中心）。sp25 OBJECT_POINTS =
  //   (0,0,0.827),(0,0.127,0.700),(0,0,0.573),(0,-0.127,0.700)
  // -> 相对扇叶中心 (0,0,0.7) 为 (0,0,±h),(0,±h,0)，h = 0.127
  double blade_half_size = 0.127;
  // 模拟器给的枪口补偿（POWER_RUNE_INTERFACE.md:105）：
  //   枪口相对相机光心，云台系(X前 Y左 Z上)：前 73.83mm、右 1.08mm、下 28.48mm
  bool use_muzzle_offset = true;
  double muzzle_offset[3] = {0.07383008, -0.00108151, -0.02848256};
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1303.686, 0, 720, 0, 1303.686, 540, 0, 0, 1);
  cv::Mat distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);

  // HW26 自身参数
  int buff_mode = 1;    // 1 = 小符, 2 = 大符
  int buff_color = 0;   // 要打的符颜色: 0 = BLUE, 1 = RED
  float confidence_threshold = 0.5f;
  float nms_threshold = 0.4f;
  // 是否启用 HW26 Selector 的两道过滤（关掉可用于诊断"检出但被过滤"）
  bool filter_color = true;
  bool filter_activation = true;

  // 弹道 / 瞄准
  double bullet_speed = 22.0;      // 模拟器弹速
  double gravity = 9.81;
  double yaw_offset_deg = 0.0;
  double pitch_offset_deg = -20.0; // 抵消枪管固定 20° 上仰
  double predict_time = 0.10;      // 额外提前量（秒）
  double gimbal_lead_time = 0.0;   // 手动补偿云台稳态滞后（秒）
  double alg_latency = 0.0;        // 若 >0 则用它，否则用实测 now_gap

  // 观测质量闸
  double min_distance = 2.0;
  double max_distance = 15.0;

  // 火控
  bool fire_enabled = true;
  double fire_err_rad = 0.02;
  int fire_stable_frames = 3;
  double fire_cooldown = 1.0;

  RefFrame frame = RefFrame::kFace;
  // 符面法向的平滑系数（面系固定不动才好；符本身不动，所以可以很慢）
  double face_ema = 0.10;         // 前 face_lock_frames 帧
  double face_ema_locked = 0.003; // 之后
  int face_lock_frames = 120;
  // ===== 以下四个护栏直接对齐 ④ sp25_buff/cpp_lib/buff_vision.cpp:190-200 =====
  // ④ 能在模拟器上跑稳，主要就是靠这四个；缺一个就会把 PnP/滤波噪声放大成晃动。
  double max_slew_deg_s = 40.0;   // ④ SP25_BUFF_LOOP_MAX_SLEW_DEG（rad/s × dt，帧率无关）
  double cmd_smooth_alpha = 0.5;  // ④ SP25_BUFF_LOOP_SMOOTH（指令低通）
  double max_aim_step_px = 300.0; // ④ SP25_BUFF_LOOP_MAX_AIM_STEP_PX（aim 瞬移拒帧）
  double stale_ms = 200.0;        // ④ SP25_BUFF_STALE_MS
  double max_cmd_step_deg = 0.0;  // 额外的硬限幅，默认关（已被 slew 限幅覆盖）
  // 是否启用 sp25 的 R 中心精修（二值化100+5x5膨胀+圆mask+minAreaRect）
  bool refine_r_center = true;
  bool visualize = true;
  int log_every = 30;
  std::string log_path;  // 空则不打日志
};

struct Output {
  bool control = false;
  bool fire = false;
  double yaw = 0.0;    // rad, 外部系
  double pitch = 0.0;  // rad, 外部系（负=抬枪）
  int aim_px_x = -1;
  int aim_px_y = -1;
  bool solved = false;
  bool quality = false;
  double buff_dis = -1.0;
  int n_det = 0;        // 本帧网络检出数
  int n_selected = 0;
  int status = 3;       // HW26 StatusType: 0 CONVERGING,1 TRACKING,2 TEMP_LOST,3 LOST
  double roll = 0.0;
  double roll_velocity = 0.0;
  double fire_err = -1.0;
  double latency_ms = 0.0;
  double lead_s = 0.0;
  double fly_s = 0.0;
  float max_score = 0.0f;          // 本帧全候选的最大类别分（不设阈值），用于选 confidence_threshold
  std::vector<DetInfo> dets;       // 本帧所有检出的类别明细
  // 可视化：两个点，直径相同
  //   绿点 = 未来打击点（预测命中点）
  //   蓝点 = 当前枪管射线与"整个符所在平面"的交点（含枪管 20° 上仰与枪口平移补偿）
  int green_px_x = -1, green_px_y = -1;
  int blue_px_x = -1, blue_px_y = -1;
  bool blue_valid = false;
  double blue_range = -1.0;   // 枪管交点到枪口的距离（诊断用）
  double face_ex = 0.0;       // 面系下径向矢量出平面分量 |e.x|（应≈0，验证面系成立）
  double face_tilt_deg = 0.0;
  double reproj_px = -1.0;
  int n_solutions = 0;        // IPPE 给出的解个数
  int sol_pick = -1;          // 本帧选了哪一个
  double cmd_jump_deg = 0.0;
  bool cmd_limited = false;
  bool aim_jumped = false;        // 本帧是否因 aim 瞬移被拒（④ 同款护栏）
  double fire_aim_yaw = 0.0;      // 开火闸用的原始弹道解
  double fire_aim_pitch = 0.0;   // 本帧指令是否被速率限制夹过  // 与上次有效指令的角度跳变（抖动指标）    // 自检：PnP 四角重投影 vs 检测关键点的最大误差 // 面法向与相机光轴夹角（= 视角偏离正对的程度）
  cv::Mat debug;  // 仅 cfg.visualize 时填充
};

class Pipeline {
 public:
  explicit Pipeline(const Config &cfg);
  ~Pipeline();

  // 每帧调用。rgb 为 RGB888 行优先（模拟器 BIMG 的格式）。
  Output process(const uint8_t *rgb, int width, int height, int64_t sim_timestamp_ns,
                 float gimbal_yaw, float gimbal_pitch, float gimbal_roll);

  // 自检：对单张 BGR 图跑一次检测，返回检测数（不做跟踪/瞄准）
  int self_test_image(const cv::Mat &bgr);

  // 合成自检用：直接喂 6 个关键点（图像坐标），绕过检测器。传空则恢复用检测器。
  void set_external_kpts(const std::vector<cv::Point2f> &kpts);
  void clear_external_kpts();

  const Config &config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// 弹道：无阻力抛物线低伸解。返回 false 表示无解。
// d 水平距离(m), h 相对高度(m), v0 初速(m/s)
bool solve_ballistic_no_drag(double d, double h, double v0, double g,
                             double *pitch_up_rad, double *fly_time_s);

}  // namespace hw26_bridge
