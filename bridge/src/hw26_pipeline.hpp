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
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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

// ─────────────────────────────────────────────────────────────────────────────
// 相机 -> 云台 的外参：**全桥唯一来源**
//
// 模拟器的相机是"只取 CAM_DIRECTION 的位置、朝向跟随云台"的，所以相机相对云台的
// 姿态必须由本桥自己知道（模拟器的 rt/gimbal_pose 只发布云台姿态）。
//
// 当前模拟器把相机装成与【枪管平行】：光轴比云台前向高 20.00°
// （= SHOT_DIRECTION 的仰角；枪管同样比云台前向高 20.00°，两者夹角 0°）。
//
// 历史值对照（同一含义、只换挂装）：
//   [0,0,1, -1,0,0, 0,-1,0]                相机在云台轴上（原模拟器，比枪管低 20°）
//   [0,0.342020,0.939693, -1,0,0, 0,-0.939693,0.342020]  与枪管平行（现在）
// 改这里一处即可：gimbal_to_camera() 与所有调用点都由它派生。
inline Eigen::Matrix3f R_camera2gimbal()
{
  Eigen::Matrix3f m;
  m << 0.f, 0.342020f, 0.939693f,
      -1.f, 0.f, 0.f,
      0.f, -0.939693f, 0.342020f;
  return m;
}

/// 云台系 -> 相机系（= R_camera2gimbal 的转置；不要再手写公式）
inline Eigen::Vector3f gimbal_to_camera(const Eigen::Vector3f & p_gimbal)
{
  return R_camera2gimbal().transpose() * p_gimbal;
}

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
  kHw26,  // HWauto 自带的 YOLO11s-pose（9 关键点，分类颜色 + 激活）—— 目前唯一后端
};

// 单个检出的类别摘要（用于诊断"检出但被过滤"）
struct DetInfo {
  int color = -1;   // 0=BLUE 1=RED
  int label = -1;   // 0=未激活 1=已激活
  float confidence = 0.0f;
  float cx = 0.0f, cy = 0.0f;
};

struct Config {
  // 检测后端。**不是从 yaml 读的**：由 Pipeline 构造函数按模型的输出形状自动判定，
  // 所以配置里只写一个 model: 路径，不存在"开关与路径不匹配"的可能。
  //   [1, 5040, 26] -> kHw26（9 关键点，分类 颜色 + 激活）
  DetectorKind detector = DetectorKind::kHw26;
  // 模型路径（yaml 的 model:）
  std::string model_path;
  std::string device = "CPU";   // OpenVINO 设备名: CPU / GPU / AUTO / NPU
  // 打击板四点的半边长（相对扇叶中心），(0,0,±h),(0,±h,0)。
  // 走桥自实现的 PnP，不调用 HW26 的 Solver（它的 BUFF_POINTS=0.114 用在模拟器上
  // 会让距离偏大约 10%）。
  // blade_half_size 是"四角点"那一批物理点的值（历史/sp25 口径，当前无后端使用）。
  double blade_half_size = 0.127;
  // HW26 后端的 t/l/b/r 是八边形相邻顶点的【中点】，是另一批物理点，半径不同。
  // 用两片靶标的图像间距反推真值距离得 0.127 * 6.4/9.45 ≈ 0.086，取 0.088。
  // 实测：用 0.127 时 dis≈9.4 m，用 0.088 时 dis≈6.5 m（真值≈6.4 m）—— 已验证。
  // 注意 reproj 查不出这个尺度错误：平面靶标 PnP 有尺度不可分辨性。
  double blade_half_size_hw26 = 0.088;
  // 模拟器给的枪口补偿（POWER_RUNE_INTERFACE.md:105）：
  //   枪口相对相机光心，云台系(X前 Y左 Z上)：前 73.83mm、右 1.08mm、下 28.48mm
  bool use_muzzle_offset = true;
  double muzzle_offset[3] = {0.07383008, -0.00108151, -0.02848256};
  // 对齐实车 CS016 + 6mm 镜头: fx = fy = 6.00mm / 3.45um = 1739.1（fov_y = 34.5°）
  // 原 1303.686 对应 fov_y = 45°，会让 PnP 把 6.9m 的符算成 9.2m
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1739.1, 0, 720, 0, 1739.1, 540, 0, 0, 1);
  cv::Mat distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);

  // HW26 自身参数
  int buff_mode = 1;    // 1 = 小符, 2 = 大符
  int buff_color = 0;   // 要打的符颜色: 0 = BLUE, 1 = RED
  float confidence_threshold = 0.5f;
  int num_reverse = 0;   // 0=编号 1=A(相位靠前,先打的那片)；1=两组编号对调
  float nms_threshold = 0.4f;
  // 是否启用 HW26 Selector 的两道过滤（关掉可用于诊断"检出但被过滤"）
  bool filter_color = true;
  bool filter_activation = true;
  // 双靶标（大符专用）：大符同时高亮 2 个靶标，HW26 的 selector 只会保留 1 个，
  // 而两靶标在转盘上交替"离光轴更近" -> 相位每帧跳 72° -> 超过 switch_buff_angle
  // (45.8°) 的跳变判据 -> 该帧被丢弃 -> RANSAC 永远拿不到数据 -> st 卡在 CONVERGING。
  // 打开后桥自己保留两个候选，并利用"它们刚性固连在同一个转盘上、相位差是常数
  // (72° 的整数倍)"这一点，把两次观测融合成同一个三角函数体系下的一次观测。
  bool dual_target = true;
  // 弦相位（实验）：用两片靶心的连线方向当相位，理论上与圆心无关。
  // 实测（2026-10-12 23:08 真实运行）反而更差：ch=1 帧的超物理步占比 32.6%
  // vs 圆心路径 24.2%，因为两片位置各自带 IPPE 双解误差，做差并不会抵消离散翻转。
  // 默认关闭，保留代码供参考。
  bool chord_phase = false;
  // 两片 PnP roll 融合（本步实验）：A/B 是同一转轴相位的两次独立测量，
  // 刚性固连 -> roll_B = roll_A + Δ (Δ=k·72°)。把 B 对齐 A 后平均，噪声降 sqrt(2)，
  // 且【完全不用位置/圆心】，不会受圆心换支造成的"持续相移"影响（实测 +131°）。
  bool phase_fuse_dual = true;
  // 相位滑动平均长度（低通）。原因：HW26 内部用【相邻两帧差分】算角速度
  // (v = droll/dt)，相位测量噪声 ≈3 度/帧 而真实变化才 ≈2.9 度/帧，
  // 信噪比≈1 -> v 的噪声 ~1.5 rad/s（信号才 ~2 rad/s）-> RANSAC 内点数
  // 只有 ~15，够不到 min_inliers=35，st 永远停在 0。
  // 先平滑再喂，噪声降 sqrt(N)。N=6 时预计内点数 ~34~45。
  // 代价：滞后 (N-1)/2 帧，预测时要补回去（见 base_lead）。
  int phase_ma_n = 6;
  // 大符用桥自己的【局部相位 tracker】做短时外推（见 src/local_tracker.hpp）。
  // 打开后：即使 HW26 的 tracker 还没到 TRACKING（st=0），只要局部拟合够样本
  // （默认 6 个，约 0.2~0.3 s）就能给出预测命中点。HW26 到 TRACKING 后仍优先用它。
  int local_tracker = 1;
  double local_rms_gate_deg = 2.5;  // 局部估计器的 rms 门（含重锁门）；调大可让脏数据也出预测
  double local_coast_s = 0.2;       // 灭灯多久之内允许滑行；超过就清空参数（真灭灯）
  int local_model = 0;   // 0=csine 1=AR 2=EKF 3=free 4=AR+包络 5=裸AR 6=查表+滑行

  // ---- 转盘圆心 C 的来源 ----
  // 圆心 C 是"相位 → 打击点"的唯一基准：roll = atan2(t.z-C.z, C.y-t.y)，
  // 预测点 = (C.x, C.y - R cosθ, C.z + R sinθ)。所以 C 错了，一切都跟着错。
  //   0 = dual：由【两片位置 + 半径 0.7 硬约束】联立解中垂线交点（原来的做法）。
  //             好处是第一帧就有；坏处是另一片的位置会进到这一片的结算里 ——
  //             实测两片的面外间距（= 诊断量 rcd = |t0.x - t1.x|）可达 0.60 m。
  //   1 = single_history：只用一片的【历史 t】做"已知半径 0.7"的圆拟合。
  //             彻底切断两片互相污染，并能把该片自己 IPPE 翻转那几帧当外点剔掉。
  //             代价：要攒够 center_hist_min 个样本才建得起来；
  //             在此之前自动回退到 0（dual），所以不会比原来差。
  //   （曾经用过的"由 PnP 朝向反推"已废弃：四个物点构成正方形，绕法向有 4 重退化，
  //     实测分歧 0.8~1.6 m，所以这里不再提供这一档。）
  int center_source = 0;
  int center_hist_min = 8;            // 圆拟合所需最少样本数
  double center_hist_span_m = 0.05;   // 样本在面内的最小跨度（弧太短 -> 法方程病态）
  double center_hist_gate_m = 0.15;   // 外点门：||t-C| - 0.7| 超过它就当外点剔除
  double center_hist_keep_s = 8.0;    // 历史只保留最近这么久（圆心静止，限内存而已）

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
  // 开火纪律（针对"空单"）：模拟器冷却 1.0 s 且 fire 被消费，桥若每帧都发
  // fire=true，实际开火时刻就由模拟器冷却到期决定，桥的对准判定形同虚设。
  // 实测开火命令 97% 间隔 <0.2s、开火时 status 有 261 帧是 CONVERGING(0)。
  // 现在只在【预测可信】时允许开火：本地模型 ready、rms 小、样本够。
  // 命中后换组时模型会短暂失锁 -> 自动停火，等重新锁定再打（正好治"打中了又打"）。
  // 瞄准预测来源：本地 mode7 模型 ready 时【优先用它】，不要被 HW26 的 predictor
  // 接管。实测 control=1 的 682 帧里 438 帧走了 HW26（status=TRACKING），
  // aim 跳变 p90=136px、12% >100px —— 紫点绕符心转圈的主因。
  // aim_local_only=1 时：本地未 ready 就不出瞄准（宁可不瞄，不转圈）。
  bool aim_local_only = true;
  bool fire_require_lock = true;
  double fire_rms_gate_deg = 15.0;
  int fire_min_n = 15;

  RefFrame frame = RefFrame::kFace;
  // 符面法向的平滑系数（面系固定不动才好；符本身不动，所以可以很慢）
  double face_ema = 0.10;         // 前 face_lock_frames 帧
  double face_ema_locked = 0.003; // 之后
  int face_lock_frames = 120;
  // ④ 能在模拟器上跑稳，主要就是靠这四个；缺一个就会把 PnP/滤波噪声放大成晃动。
  double max_slew_deg_s = 40.0;
  double cmd_smooth_alpha = 0.5;
  double max_aim_step_px = 300.0;
  double stale_ms = 200.0;
  bool visualize = true;
  int log_every = 1;   // CSV 文件：必须 1（每帧一行）
  int console_every = 10;   // 屏幕打印：每 N 帧一行（不影响 CSV）
  int visualize_every = 1;  // 显示节流：1 = 每帧都显示
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
  float max_score = 0.0f;          // 本帧全候选的最大类别分（不设阈值），用于选 confidence_threshold
  std::vector<DetInfo> dets;       // 本帧所有检出的类别明细
  // 可视化：两个点
  //   紫点（原绿，BGR 240,32,160）= 未来打击点（预测命中点）
  //   蓝点 = 当前枪管射线与"整个符所在平面"的交点（含枪管 20° 上仰与枪口平移补偿）
  //          数据仍计算并打印，但已不再绘制
  int green_px_x = -1, green_px_y = -1;
  int blue_px_x = -1, blue_px_y = -1;
  double face_ex = 0.0;       // 面系下径向矢量的出平面分量 |e.x|（应≈0，验证面系成立）
  double face_tilt_deg = 0.0; // 面法向与相机光轴的夹角（视角偏离正对的程度）
  double reproj_px = -1.0;    // 自检：PnP 四角重投影 vs 检测关键点的最大误差（当前未计算）
  // 双靶标诊断
  int dual_targets = 0;         // 本帧融入了几个靶标（大符正常 = 2）
  double dual_delta_deg = 0.0;  // 吸附后的相位差 Δ（应为 72 或 144 度）
  double dual_resid_deg = -1.0; // 吸附残差 |d − k·72°|：应为几度。大了说明两片
                                // 并不是隔着整数个槽位，"刚性固连"的假设不成立
  double dual_rcd_m = -1.0;     // 两片各自解出的 R_center 分歧：同一转盘应只有几 cm。
                                // 大了说明其中一片的位姿不自洽 -> 不能融合
  // 圆心 C 的来源与拟合质量（center_source=1 时才有意义）
  //   center_src: 0 = 两片联立（dual）, 1 = 单片历史圆拟合
  //   center_n  : 圆拟合实际用到的样本数（剔除外点后）
  //   center_rms_m: 圆拟合残差 rms（米）。应 ~1e-2；大了说明样本脏或跨度不足
  int center_src = 0;
  int center_n = 0;
  double center_rms_m = -1.0;
  // 诊断（不依赖身份）：本帧两片 roll 与上一帧两片 roll 做【最优配对】后的两个残差。
  // 某一对接近 180 度 = 那片发生了 IPPE 双解翻转（相位被翻到对面去了）。
  double droll_deg[2] = {0.0, 0.0};
  int sol_pick_t[2] = {-1, -1};  // 两片各自选了第几个 IPPE 解
  // 两片板心在符面内的距离（弦长）。这是【不依赖任何朝向】的独立观测量：
  //   相邻槽位(72 度) 应为 2*0.7*sin(36) = 0.823 m
  //   隔一槽位(144 度) 应为 2*0.7*sin(72) = 1.331 m
  // 它与 dlt 是否一致，是判断"两片确实固连在同一转盘上"的直接证据。
  double chord_m = -1.0;
  // ---- 组号/身份诊断（定位「黄/品红为什么会换」）----
  int dbg_pose0_is_a = -1;   // A/B 判定：1 = poses[0] 是 A
  int dbg_src_a = -1;        // 组 0（黄）来自哪个检出下标
  int dbg_src_b = -1;        // 组 1（品红）来自哪个检出下标
  double dbg_conf0 = -1.0;   // selected 顺序下两个检出的置信度
  double dbg_conf1 = -1.0;
  double dbg_g0x = -1.0;     // 组 0 关键点质心（图像像素）
  double dbg_g0y = -1.0;
  double dbg_g1x = -1.0;
  double dbg_g1y = -1.0;
  double dbg_roll_a = 0.0;   // 融合前两片各自的原始相位
  double dbg_roll_b = 0.0;
  double dbg_rca_m = -1.0;   // 两片各自解出的 R_center 距离
  double dbg_rcb_m = -1.0;
  double cand_px[2] = {-1.0, -1.0};
  // 桥自己的局部相位 tracker 的诊断：样本数 / 是否 ready / 角速度 / 角加速度
  int local_n = 0;
  int local_ready = 0;
  double local_w = 0.0;
  int local_locked = 0;      // 正弦参数是否已锁存
  double local_omega = 0.0;   // 拟合出的 omega（应落在 1.884~2.000）
  double local_amp = 0.0;     // 拟合出的角速度振幅 a（应落在 0.780~1.045）
  double local_rms_deg = -1.0;
  double local_span_deg = 0.0;
  int local_drop = 0;     // 因跳变被丢掉的帧数（只丢帧、不清窗）
  int local_lost = 0;
  int local_hit = 0;      // 连续 >= hit_s 秒无空窗 -> 判定命中
  double local_theta = 0.0;  // 本帧实际喂进局部 tracker 的相位（rad）
  int local_chord = 0;       // 2=两片roll融合/roll约定 1=弦相位 0=圆心路径
  int local_phys_rej = 0;    // mode 7 被"模型锚定创新门"拒绝的帧数（累计）
  // ---- 可视化用（全部在【车体系】，方便直接喂 project_px）----
  // 关键：两片的实测板心要【分开】留着，否则看不出是哪一片坏了/被平均了
  bool meas_a_valid = false;
  bool meas_b_valid = false;
  Eigen::Vector3f meas_a_ch = Eigen::Vector3f::Zero();  // A 片实测板心
  Eigen::Vector3f meas_b_ch = Eigen::Vector3f::Zero();  // B 片实测板心
  // 每个检出各自的关键点（可视化用）。以前只画 detected[0]，而 detected[0] 是
  // 置信度最高的那个，两片靶标的置信度会互相超过 -> 画出来的点在两片之间来回跳。
  // 注意这只影响"画什么"，不影响 tracker 的输入（tracker 用的是 A 片的相位）。
  std::vector<std::vector<cv::Point2f>> det_kpts;
  // |R_center|：到【转盘中心】的距离。它与转盘的相位无关（转盘中心是固定点），
  // 所以是标定 blade_half_size 的理想判据：
  //     h_true = blade_half_size * rc_true / rc_measured
  // （reproj 查不出尺度错误：平面靶标的 PnP 有尺度不可分辨性）
  double rc_dist_m = -1.0;
  int n_solutions = 0;        // IPPE 给出的解个数
  int sol_pick = -1;          // 本帧选了哪一个
  double cmd_jump_deg = 0.0;      // 与上次有效指令的角度跳变（抖动指标）
  double fire_aim_yaw = 0.0;      // 开火闸用的原始弹道解
  double fire_aim_pitch = 0.0;
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

// 固定半径圆的圆心拟合（Config::center_source == 1 时用；定义在 hw26_pipeline.cpp）。
//   hist      : (t_rel, t_chassis) 的历史；t_chassis 是【车体系】位置
//   Rc2f      : 车体系 -> 符面系 的旋转
//   radius    : 硬约束半径（= consts::BUFF_RADIUS = 0.7）
//   min_n / min_span_m : 最少样本数、样本在面内的最小跨度（太短则法方程病态）
//   gate_m    : 外点门 ||t-C| - radius|，超出的样本剔除（模拟 IPPE 翻转那几帧）
//   t_now/keep_s: 只保留最近 keep_s 秒的样本
// 成功返回 true，并写出圆心（符面系）、实际参与拟合的样本数、残差 rms（米）。
// 只对 (C_y, C_z, k) 做线性最小二乘，所以没有"中垂线两支"的二义性。
bool fit_fixed_radius_center(
    const std::deque<std::pair<double, Eigen::Vector3f>> &hist, const Eigen::Matrix3f &Rc2f,
    double radius, int min_n, double min_span_m, double gate_m, double t_now, double keep_s,
    Eigen::Vector3f *c_out, int *n_out, double *rms_out);

}  // namespace hw26_bridge
