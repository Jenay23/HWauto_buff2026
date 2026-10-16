#pragma once

// Pipeline 的 pimpl 状态定义（P2-7：从 hw26_pipeline.cpp 原样搬出）。
//
// 它是**私有头**：只有管线的 .cpp 文件包含它。存在的理由很单纯 ——
// 分节函数是 Pipeline::Impl 的 static 成员，Impl 的定义可见了，
// 它们才能被定义到别的编译单元里（P2-8 起）。

#include <chrono>
#include <deque>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

#include "hw26_pipeline.hpp"
#include "local_tracker.hpp"
#include "buff_algo/detector/detector_common.hpp"   // DetectedBuff

namespace hw26_bridge {

using buff_algo::DetectedBuff;   // 与 hw26_pipeline.cpp 里同一套 using（重复声明无害）

struct Pipeline::Impl {
  Config cfg;
  ov::Core core;
  ov::CompiledModel compiled;
  ov::InferRequest req;
  cv::Size in_size{640, 384};
  size_t in_elems = 0;
  buff_algo::BuffDetectorConfig dcfg;

  // 符面系状态（车体系下）
  bool face_ready = false;
  Eigen::Vector3f n_face = Eigen::Vector3f::UnitX();  // 面法向（车体系，平滑后）
  Eigen::Matrix3f R_chassis_face = Eigen::Matrix3f::Identity();
  Eigen::Vector3f c_face_chassis = Eigen::Vector3f::Zero();  // 符心（车体系，平滑后）
  int face_frames = 0;

  double prev_cmd_yaw = 0.0, prev_cmd_pitch = 0.0;
  bool have_prev_aim = false;
  double prev_cmd_out_yaw = 0.0, prev_cmd_out_pitch = 0.0;
  bool have_prev_cmd_out = false;
  cv::Point2f prev_aim_px{0.f, 0.f};
  bool have_prev_aim_px = false;
  double prev_t_rel = 0.0;
  Eigen::Vector3f aim_ch_abs = Eigen::Vector3f::Zero();  // 未来打击点（车体系，绝对）
  bool aim_ch_valid = false;
  std::vector<cv::Point2f> last_kpt;   // 本帧检测到的关键点（可视化）
  bool have_rcenter = false;
  bool have_rcenter_rough = false;
  cv::Point2f last_rcenter{0.f, 0.f};
  cv::Point2f last_rcenter_rough{0.f, 0.f};

  // IPPE 双解消歧用的"上一帧位姿"。大符每帧要解两片，所以这里存【上一帧的全部位姿】
  // （最多 2 个）。
  //
  // 之前试过"按 selected 的下标分槽"，那是错的：selected 是按【置信度】排序的，
  // 两片置信度一换（实测 0.91 <-> 0.89），槽位就交叉了，连续性状态仍然对错了片。
  //
  // 正确做法是【不依赖身份】：每个靶标的两个 IPPE 解，各自与上一帧的【所有】位姿
  // 比连续性，取最连续的那个。上一帧的两个位姿里必然有一个是它的（另一片是噪点，
  // 距离远），所以 min-over-all 一定选对。
  std::vector<Eigen::Vector3f> prev_ts;
  std::vector<Eigen::Quaternionf> prev_qs;
  std::vector<float> prev_rolls;   // 上一帧两片各自的 roll（诊断用）
  // 喂给 tracker 之前的相位滑动平均（低通）。存【解缠后】的相位，避免 2pi 处的假跳变。
  std::deque<double> roll_hist;
  double last_roll_raw = 0.0;
  double last_roll_unwrapped = 0.0;
  bool have_last_roll = false;
  // 帧间隔（用于把滤波滞后换算成时间）
  double prev_frame_t = 0.0;
  bool have_prev_frame_t = false;
  double dt_frame_ema = 0.033;
  LocalRuneTracker local_trk;   // 桥自己的局部相位 tracker（大符）
  // 弦相位：两片靶心之差在符面内的方向角。用来给局部 tracker 喂相位（与圆心无关）。
  double last_local_theta = 0.0;
  bool have_last_local_theta = false;
  // 由两片位置联合解出的圆心（符面系）。圆心是固定点，所以它的连续性极可靠，
  // 可以用来在"中垂线两支解"里选正确的那支。
  Eigen::Vector3f center_face = Eigen::Vector3f::Zero();
  bool have_center = false;
  // 最近一次"通过位置内点门"的橙点相对圆心的面系偏移。瞄准用它（保持单一约定），
  // 门外的帧沿用上一次的值，避免"橙点锚定 <-> 圆上取点"来回切造成的大跳。
  Eigen::Vector3f orange_rel = Eigen::Vector3f::Zero();
  bool orange_rel_valid = false;
  // center_source=1 时用来解圆心 C 的历史：只存【被选中的那一片】的 t（车体系）。
  // 为什么身份切换不要紧：两片点亮的靶标都落在【同一个半径 0.7 的圆】上，
  // 所以即使帧间"谁被选中"换了，这些点仍在同一个圆上 —— 拟合对身份免疫。
  std::deque<std::pair<double, Eigen::Vector3f>> center_hist;  // (t_rel, t_chassis)
  int center_fit_n = 0;            // 拟合实际用到的样本数（剔除外点后）
  double center_fit_rms_m = -1.0;  // 拟合残差 rms（米）
  // 依据：相位对圆心敏感度 1/0.7 ≈ 1.43 rad/m（圆心差 0.1m -> 相位差 8 度），
  // 而圆心是场上的固定点（激活内不动、车也不动），不该每帧重解。

  std::unique_ptr<buff_algo::BuffSelector> selector;
  std::unique_ptr<buff_algo::BuffPnPSolver> solver;
  std::unique_ptr<buff_algo::BuffTracker> tracker;
  std::unique_ptr<buff_algo::BuffPredictor> predictor;

  int64_t anchor_ns = 0;
  bool have_anchor = false;
  int64_t prev_sim_ns = 0;
  int frame_count = 0;
  double last_fire_t = -1e9;
  int align_run = 0;
  /// 锁存的旋向（+1 = 相位递增，-1 = 递减，0 = 还没判出来）。只用来决定
  /// A/橙 是谁，不参与任何滤波/预测。
  double dir_sgn = 0.0;
  /// 锁定期到什么时候（图像时间轴）；判准一次后到点前不再重新判。
  double dir_lock_until_t = -1e18;
  double last_latency_ms = 0.0;
  std::ofstream log;
  cv::FileStorage tracker_fs;

  explicit Impl(const Config &c) : cfg(c) {}

  // ---- 分节函数（P2：从 process() 原样搬出，逻辑一行未改）----
  static std::vector<DetectedBuff> detect_blades(Impl &I, const cv::Mat &bgr, Output &out);
  static void log_frame(Impl &I, Output &out, const Eigen::Matrix3f &R_cg,
                        const Eigen::Vector3f &t_muzzle, double lead, double fly,
                        int64_t sim_timestamp_ns,
                        std::chrono::steady_clock::time_point t_start, float gimbal_yaw,
                        float gimbal_pitch, float gimbal_roll);
};

}  // namespace hw26_bridge
