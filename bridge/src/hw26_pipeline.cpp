#include "hw26_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "buff_algo/common/types.hpp"
#include "detector_sp25.hpp"

namespace hw26_bridge {

using buff_algo::Buff;
using buff_algo::BuffDetection;
using buff_algo::BuffMode;
using buff_algo::BuffState;
using buff_algo::DetectedBuff;
using buff_algo::LetterboxInfo;
using buff_algo::Pose3f;
using buff_algo::StatusType;

namespace {

// ---- 统一的投影助手（消除轴向歧义）----
// 云台系(X前 Y左 Z上) -> 相机系(OpenCV: X右 Y下 Z前)
// 委托给 hw26_pipeline.hpp 里的单一来源 gimbal_to_camera()，
// 不要再在这里手写公式——相机挂装一变这里就会静默错。
inline Eigen::Vector3f gimbal_to_cam(const Eigen::Vector3f &p_g) {
  return gimbal_to_camera(p_g);
}
// 相机系 -> 像素（模拟器畸变为 0）。depth = p_cam.z()
inline bool project_cam(const Eigen::Vector3f &p_cam, const cv::Mat &K, int *px, int *py) {
  if (p_cam.z() < 0.2f) return false;
  const double Z = static_cast<double>(p_cam.z());
  *px = static_cast<int>(std::lround(K.at<double>(0, 2) + K.at<double>(0, 0) * p_cam.x() / Z));
  *py = static_cast<int>(std::lround(K.at<double>(1, 2) + K.at<double>(1, 1) * p_cam.y() / Z));
  return true;
}

// 复刻 sp25 的 Buff_Detector::get_r_center（buff_detector.cpp:9-76）：
//   粗估点 + 二值化(灰>100) + 5x5 膨胀 + 圆形 mask + minAreaRect 选最方正且离粗估最近的轮廓
cv::Point2f refine_r_center(const cv::Mat &bgr, const cv::Point2f &rough,
                            const cv::Point2f &blade_bottom, const cv::Point2f &blade_center) {
  cv::Mat gray, binary, dilated;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY);
  cv::dilate(binary, dilated, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}), cv::Point(-1, -1),
             1);
  const double radius = std::max(4.0, cv::norm(blade_bottom - blade_center) * 0.8);
  cv::Mat mask = cv::Mat::zeros(dilated.size(), CV_8U);
  cv::circle(mask, rough, static_cast<int>(radius), cv::Scalar(255), cv::FILLED);
  cv::bitwise_and(dilated, mask, dilated);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  cv::Point2f best = rough;
  double best_ratio = 1e18;
  for (const auto &c : contours) {
    const cv::RotatedRect rr = cv::minAreaRect(c);
    if (rr.size.width < 1.f || rr.size.height < 1.f) continue;
    const double mx = std::max(rr.size.width, rr.size.height);
    const double mn = std::min(rr.size.width, rr.size.height);
    const double ratio =
        mx / mn + cv::norm(rr.center - rough) / std::max(1.0, radius / 3.0);
    if (ratio < best_ratio) {
      best_ratio = ratio;
      best = rr.center;
    }
  }
  return best;
}

constexpr int64_t kCandidateCount = 5040;
constexpr int64_t kOutputColumns = 26;  // 4 + 4 + 9*2
// sp25 的 yolo11_buff_int8：[1, 17, 8400]，17 = 4 bbox + 1 score + 6*2 kpt。
// 用它把"配置里只写一个 model: 路径"变成可能：后端由输出形状判定。
constexpr int64_t kSp25OutRows = 17;
constexpr double kSimPitchLimit = 0.785;

double wrap_pi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

// 与 ④ sp25_buff 桥一致的云台姿态 -> 旋转矩阵（外部系 X前 Y左 Z上，ZYX 顺序）
Eigen::Matrix3f gimbal_rotation(float yaw, float pitch, float roll) {
  Eigen::Matrix3f R =
      (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
       Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()))
          .toRotationMatrix();
  return R;
}

// 内嵌默认 tracker 参数（取自 HWauto_buff2026/config/tracker/tracker.yaml）
const char *kDefaultTrackerYaml = R"YAML(%YAML:1.0
---
buff_tracker:
  small_buff_observer:
    switch_buff_angle: 45.8
    R_center_filter_ratio: 0.1
    kf_roll:
      process_noise: !!opencv-matrix
        rows: 2
        cols: 2
        dt: f
        data: [ 1e-1, 0, 0, 1e+0 ]
      measurement_noise: !!opencv-matrix
        rows: 1
        cols: 1
        dt: f
        data: [ 1e-1 ]
    kf_yaw:
      process_noise: !!opencv-matrix
        rows: 2
        cols: 2
        dt: f
        data: [ 1e-1, 0, 0, 1e-0 ]
      measurement_noise: !!opencv-matrix
        rows: 1
        cols: 1
        dt: f
        data: [ 1e-1 ]
  big_buff_observer:
    switch_buff_angle: 45.8
    R_center_filter_ratio: 0.1
    ransac:
      max_iterations: 200
      threshold: 0.5
      min_omega: 1.884
      max_omega: 2.0
      min_amplitude: 0.780
      max_amplitude: 1.045
      min_inliers: 100
      max_abs_speed: 2.090
    kf_yaw:
      process_noise: !!opencv-matrix
        rows: 2
        cols: 2
        dt: f
        data: [ 1e-1, 0, 0, 1e-0 ]
      measurement_noise: !!opencv-matrix
        rows: 1
        cols: 1
        dt: f
        data: [ 1e-1 ]
  small_buff_status:
    max_temp_lost_frames: 20
    max_converging_frames: 20
  big_buff_status:
    max_temp_lost_frames: 20
    max_converging_frames: 300
  temp_lost_return_frames: 5
  err_queue_size: 0
  approximate_framerate: 0
)YAML";

}  // namespace

bool solve_ballistic_no_drag(double d, double h, double v0, double g,
                             double *pitch_up_rad, double *fly_time_s) {
  if (!(d > 1e-3) || !(v0 > 1.0)) return false;
  const double v2 = v0 * v0;
  const double disc = v2 * v2 - g * (g * d * d + 2.0 * h * v2);
  if (disc < 0.0) return false;
  const double root = std::sqrt(disc);
  const double tan_theta = (v2 - root) / (g * d);  // 低伸解
  const double theta = std::atan(tan_theta);
  const double c = std::cos(theta);
  if (std::abs(c) < 1e-6) return false;
  if (pitch_up_rad) *pitch_up_rad = theta;
  if (fly_time_s) *fly_time_s = d / (v0 * c);
  return true;
}

struct Pipeline::Impl {
  Config cfg;
  ov::Core core;
  ov::CompiledModel compiled;
  ov::InferRequest req;
  cv::Size in_size{640, 384};
  size_t in_elems = 0;
  buff_algo::BuffDetectorConfig dcfg;

  std::unique_ptr<Sp25Detector> sp25_det;
  // 自实现的 IPPE + 双解连续性消歧（用 sp25 的物体点，HW26 的 BUFF_POINTS 尺寸不对）
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
  std::vector<cv::Point2f> ext_kpts;   // 合成自检用：外部喂的关键点
  std::vector<cv::Point2f> last_kpt;   // 本帧检测到的关键点（可视化）
  bool have_rcenter = false;
  bool have_rcenter_rough = false;
  cv::Point2f last_rcenter{0.f, 0.f};
  cv::Point2f last_rcenter_rough{0.f, 0.f};

  bool has_prev_pose = false;
  Eigen::Vector3f prev_t = Eigen::Vector3f::Zero();
  Eigen::Quaternionf prev_q = Eigen::Quaternionf::Identity();

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
  double last_latency_ms = 0.0;
  std::ofstream log;
  cv::FileStorage tracker_fs;

  explicit Impl(const Config &c) : cfg(c) {}
};

Pipeline::Pipeline(const Config &cfg) : impl_(new Impl(cfg)) {
  auto &I = *impl_;
  I.dcfg.confidence_threshold = cfg.confidence_threshold;
  I.dcfg.nms_threshold = cfg.nms_threshold;

  auto model = I.core.read_model(cfg.model_path);
  const ov::Shape shp = model->input().get_shape();
  if (shp.size() != 4 || shp[0] != 1 || shp[1] != 3) {
    throw std::runtime_error("hw26: model input must be [1,3,H,W]");
  }
  I.in_size = cv::Size(static_cast<int>(shp[3]), static_cast<int>(shp[2]));
  I.in_elems = shp[0] * shp[1] * shp[2] * shp[3];

  // ---- 检测后端由模型的输出形状自动判定（配置里只写一个 model:）----
  const ov::Shape oshp = model->output().get_shape();
  {
    std::ostringstream os;
    os << "[hw26] model in=" << shp[2] << "x" << shp[3] << " out=[";
    for (size_t i = 0; i < oshp.size(); ++i) os << (i ? "," : "") << oshp[i];
    os << "]";
    std::cout << os.str() << std::endl;
  }
  const int64_t out_rows = (oshp.size() == 3) ? static_cast<int64_t>(oshp[1]) : -1;
  const int64_t out_cols = (oshp.size() == 3) ? static_cast<int64_t>(oshp[2]) : -1;
  if (out_cols == kOutputColumns) {
    I.cfg.detector = DetectorKind::kHw26;
  } else if (out_rows == kSp25OutRows) {
    I.cfg.detector = DetectorKind::kSp25;
  } else {
    std::ostringstream os;
    os << "[hw26] FATAL: cannot tell which detector this model is from its output shape [";
    for (size_t i = 0; i < oshp.size(); ++i) os << (i ? "," : "") << oshp[i];
    os << "]. Expected HW26 [1," << kCandidateCount << "," << kOutputColumns
       << "] or sp25 [1," << kSp25OutRows << ",N]. Check the model: line in bridge.yaml.";
    throw std::runtime_error(os.str());
  }

  if (I.cfg.detector == DetectorKind::kSp25) {
    // Sp25Detector 自己会校验 [1,17,8400]，形状不对会抛
    I.sp25_det = std::make_unique<Sp25Detector>(cfg.model_path);
    std::cout << "[hw26] detector=sp25(" << cfg.model_path << ") in="
              << I.sp25_det->input_size().width << "x" << I.sp25_det->input_size().height
              << "  (6 keypoints, single class: no colour/activation info)" << std::endl;
  } else {
    // 只在这个后端被选中时才编译模型（sp25 模式下不再白加载 38MB 的 HW26 模型）
    I.compiled = I.core.compile_model(model, "CPU");
    I.req = I.compiled.create_infer_request();
    std::cout << "[hw26] detector=hw26(" << cfg.model_path
              << ")  (9 keypoints, colour + activation classified)" << std::endl;
  }

  I.selector = std::make_unique<buff_algo::BuffSelector>(cfg.buff_color);
  I.selector->set_camera_matrix(cfg.camera_matrix, cfg.distort_coeffs);
  I.solver = std::make_unique<buff_algo::BuffPnPSolver>();
  I.solver->set_camera_matrix(cfg.camera_matrix, cfg.distort_coeffs);

  if (!I.tracker_fs.open(kDefaultTrackerYaml, cv::FileStorage::READ | cv::FileStorage::MEMORY)) {
    throw std::runtime_error("hw26: failed to open embedded tracker yaml");
  }
  I.tracker = std::make_unique<buff_algo::BuffTracker>(I.tracker_fs["buff_tracker"]);
  I.tracker->set_mode(cfg.buff_mode == 2 ? BuffMode::BIG_BUFF : BuffMode::SMALL_BUFF);
  I.predictor = std::make_unique<buff_algo::BuffPredictor>(I.tracker_fs["buff_tracker"]);

  if (!cfg.log_path.empty()) {
    I.log.open(cfg.log_path, std::ios::out | std::ios::trunc);
    if (I.log) {
      I.log << "frame,sim_t_ns,latency_ms,gimbal_yaw_deg,gimbal_pitch_deg,n_det,n_sel,solved,"
               "quality,status,buff_dis,roll_rad,roll_vel,lead_s,fly_s,aim_x,aim_y,"
               "cmd_yaw_deg,cmd_pitch_deg,fire_err_rad,control,fire,"
               // appended (2026-10: diagnostics that were only on stdout before)
               //   ex_m            |e.x| of the face-frame radial vector, should be ~0
               //   reproj_px       max reprojection error of the PnP corners [px]
               //   gimbal_roll_deg gimbal roll (the CSV previously had yaw/pitch only)
               //   tilt_deg        angle between the face normal and the optical axis
               "ex_m,reproj_px,gimbal_roll_deg,tilt_deg\n";
    }
  }
}

Pipeline::~Pipeline() = default;

const Config &Pipeline::config() const { return impl_->cfg; }

void Pipeline::set_external_kpts(const std::vector<cv::Point2f> &kpts) { impl_->ext_kpts = kpts; }
void Pipeline::clear_external_kpts() { impl_->ext_kpts.clear(); }

int Pipeline::self_test_image(const cv::Mat &bgr) {
  auto &I = *impl_;

  // 必须按选中的后端跑，否则 --selftest 永远只测 HW26 的模型，
  // 两个后端会打印出逐位相同的"对比"结果（这个 bug 曾经存在过）。
  if (I.cfg.detector == DetectorKind::kSp25) {
    const std::optional<Sp25Detection> det = I.sp25_det->detect(bgr, I.cfg.confidence_threshold);
    if (!det) {
      std::cout << "  (sp25) no detection above threshold " << I.cfg.confidence_threshold
                << std::endl;
      return 0;
    }
    std::cout << "  (sp25) det conf=" << det->confidence << " max_score=" << det->max_score
              << " kpt0=(" << det->kpt[0].x << "," << det->kpt[0].y << ")"
              << " r_center=(" << det->r_center.x << "," << det->r_center.y << ")" << std::endl;
    return 1;
  }

  LetterboxInfo info;
  cv::Mat lb = buff_algo::letterbox_image(bgr, info, I.in_size);
  cv::Mat blob =
      cv::dnn::blobFromImage(lb, 1.0 / 255.0, I.in_size, cv::Scalar(), true, false, CV_32F);
  ov::Tensor in_t(I.compiled.input().get_element_type(), I.compiled.input().get_shape(),
                  blob.ptr<float>());
  I.req.set_input_tensor(in_t);
  I.req.infer();
  const float *out = I.req.get_output_tensor(0).data<const float>();
  auto det = buff_algo::decode_detections(out, kCandidateCount, kOutputColumns, info, I.dcfg);
  for (const auto &d : det) {
    std::cout << "  (hw26) det conf=" << d.detection.confidence << " label=" << d.detection.label
              << " color=" << d.detection.color << " kpt0=(" << d.keypoints[0].x() << ","
              << d.keypoints[0].y() << ")" << std::endl;
  }
  return static_cast<int>(det.size());
}

Output Pipeline::process(const uint8_t *rgb, int width, int height, int64_t sim_timestamp_ns,
                         float gimbal_yaw, float gimbal_pitch, float gimbal_roll) {
  auto &I = *impl_;
  const auto t_start = std::chrono::steady_clock::now();
  Output out;

  if (!I.have_anchor) {
    I.anchor_ns = sim_timestamp_ns;
    I.have_anchor = true;
    I.prev_sim_ns = sim_timestamp_ns;
  }
  // 相对时间（秒），避免 float 精度问题；HW26 的 tracker 只用差分
  const double t_rel = static_cast<double>(sim_timestamp_ns - I.anchor_ns) * 1e-9;
  const double dt_frame =
      (I.frame_count == 0) ? 0.0
                           : static_cast<double>(sim_timestamp_ns - I.prev_sim_ns) * 1e-9;
  I.prev_sim_ns = sim_timestamp_ns;

  // 1) RGB -> BGR
  cv::Mat rgb_view(height, width, CV_8UC3, const_cast<uint8_t *>(rgb));
  cv::Mat bgr;
  cv::cvtColor(rgb_view, bgr, cv::COLOR_RGB2BGR);

  // ══════════ 检测 + PnP：两条后端 ══════════
  // 共同结果：out.solved / out.n_det / out.quality / out.buff_dis / poses（云台系 Pose3f）
  std::vector<Pose3f> poses;
  const Eigen::Matrix3f R_cg = gimbal_rotation(gimbal_yaw, gimbal_pitch, gimbal_roll);
  // OpenCV 相机系(X右 Y下 Z前) -> 云台系(X前 Y左 Z上)。
  // 取值来自 hw26_pipeline.hpp 的单一来源：当前模拟器的相机与枪管平行
  // （光轴比云台前向高 20.00°）。旧值 [0,0,1,-1,0,0,0,-1,0] 假设相机在云台轴上，
  // 会让目标在云台系里整体低 20° -> 云台指令低 20° -> 打低。
  const Eigen::Matrix3f R_cam2gim = R_camera2gimbal();
  // 枪口相对相机光心（云台系）；模拟器 POWER_RUNE_INTERFACE.md:105 的实测外参
  const Eigen::Vector3f t_muzzle(static_cast<float>(I.cfg.muzzle_offset[0]),
                                 static_cast<float>(I.cfg.muzzle_offset[1]),
                                 static_cast<float>(I.cfg.muzzle_offset[2]));
  I.last_kpt.clear();
  I.have_rcenter = false;

  // ---- 统一的 PnP：两个检测后端共用 ----
  // 刻意【不】调用 HW26 的 BuffPnPSolver：它内部写死 consts::BUFF_POINTS = 0.114，
  // 那是 HW26 实车靶板的尺寸，用在模拟器上会让距离偏大约 10%（实测 10.18%，
  // 而 0.127 只有 0.18%）。两个模型的上/左/下/右四点指同一批物理点，所以共用一套
  // 物体点。双解消歧与首帧判据也集中在这一处，不要再写第二份。
  const double h_plate = I.cfg.blade_half_size;
  const std::vector<cv::Point3f> obj_plate = {
      {0.f, 0.f, static_cast<float>(h_plate)},
      {0.f, static_cast<float>(h_plate), 0.f},
      {0.f, 0.f, static_cast<float>(-h_plate)},
      {0.f, static_cast<float>(-h_plate), 0.f},
  };
  auto solve_pose_ippe = [&](const cv::Point2f &pt, const cv::Point2f &pl,
                             const cv::Point2f &pb,
                             const cv::Point2f &pr) -> std::optional<Pose3f> {
    const std::vector<cv::Point2f> img = {pt, pl, pb, pr};
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<double> errs;
    try {
      cv::solvePnPGeneric(obj_plate, img, I.cfg.camera_matrix, I.cfg.distort_coeffs, rvecs,
                          tvecs, false, cv::SOLVEPNP_IPPE, cv::noArray(), cv::noArray(), errs);
    } catch (const cv::Exception &) {
      return std::nullopt;
    }
    if (rvecs.empty()) return std::nullopt;

    // 双解消歧：首帧取重投影误差小者；此后取与上一帧更连续者
    int pick = 0;
    if (rvecs.size() == 1) {
      pick = 0;
    } else if (!I.has_prev_pose) {
      pick = (errs.size() >= 2 && errs[1] < errs[0]) ? 1 : 0;
    } else {
      // 连续性判据必须放在【静止系】里比较：这里用云台系，因为 prev 也在同一
      // 瞬间的云台系里记录，两者一致。
      double best = 1e18;
      for (std::size_t k = 0; k < rvecs.size(); ++k) {
        const Eigen::Vector3f t_c(static_cast<float>(tvecs[k].at<double>(0)),
                                  static_cast<float>(tvecs[k].at<double>(1)),
                                  static_cast<float>(tvecs[k].at<double>(2)));
        cv::Mat Rm;
        cv::Rodrigues(rvecs[k], Rm);
        Eigen::Matrix3f Re;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) Re(a, b) = static_cast<float>(Rm.at<double>(a, b));
        const Eigen::Vector3f t_g = R_cam2gim * t_c;
        const Eigen::Quaternionf q(Eigen::Matrix3f(R_cam2gim * Re * R_cam2gim.transpose()));
        const double dt = (t_g - I.prev_t).norm();
        const double dq =
            2.0 * std::acos(std::min(1.0, std::abs(static_cast<double>(q.dot(I.prev_q)))));
        const double score = 2.0 * dt + dq;
        if (score < best) {
          best = score;
          pick = static_cast<int>(k);
        }
      }
    }
    const Eigen::Vector3f t_cam(static_cast<float>(tvecs[pick].at<double>(0)),
                                static_cast<float>(tvecs[pick].at<double>(1)),
                                static_cast<float>(tvecs[pick].at<double>(2)));
    cv::Mat Rm;
    cv::Rodrigues(rvecs[pick], Rm);
    Eigen::Matrix3f Re;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) Re(a, b) = static_cast<float>(Rm.at<double>(a, b));

    Pose3f p;
    p.translation = R_cam2gim * t_cam;                    // 相机系 -> 云台系
    p.rotation = Eigen::Quaternionf(R_cam2gim * Re).normalized();
    I.has_prev_pose = true;
    I.prev_t = p.translation;   // 云台系
    I.prev_q = p.rotation;      // 云台系
    out.n_solutions = static_cast<int>(rvecs.size());
    out.sol_pick = pick;
    return p;
  };

  if (I.cfg.detector == DetectorKind::kSp25) {
    std::optional<Sp25Detection> det;
    if (I.ext_kpts.size() >= 6) {
      Sp25Detection d;
      d.confidence = 1.0f;
      d.max_score = 1.0f;
      for (int i = 0; i < 6; ++i) d.kpt[static_cast<std::size_t>(i)] = I.ext_kpts[static_cast<std::size_t>(i)];
      d.r_center = d.kpt[4] + 1.4f * (d.kpt[5] - d.kpt[4]);
      d.box = cv::Rect2f(d.kpt[0].x - 20.f, d.kpt[0].y - 20.f, 40.f, 40.f);
      det = d;
    } else {
      det = I.sp25_det->detect(bgr, I.cfg.confidence_threshold);
    }
    out.n_det = det.has_value() ? 1 : 0;
    out.max_score = det.has_value() ? det->max_score : 0.f;
    if (det) {
      DetInfo di;
      di.color = I.cfg.buff_color;  // sp25 的模型是单类，颜色靠配置给定
      di.label = 0;
      di.confidence = det->confidence;
      di.cx = det->kpt[0].x;
      di.cy = det->kpt[0].y;
      out.dets.push_back(di);
      out.n_selected = 1;
      for (const auto &q : det->kpt) I.last_kpt.push_back(q);
      out.det_kpts.push_back(I.last_kpt);  // sp25 只有一个检出，仅用于统一可视化
      I.last_rcenter = det->r_center;
      if (I.cfg.refine_r_center) {
        // blade_bottom = kpt[2]（下），blade_center 用模型 bbox 中心
        const cv::Point2f bc(det->box.x + det->box.width * 0.5f,
                             det->box.y + det->box.height * 0.5f);
        I.last_rcenter = refine_r_center(bgr, det->r_center, det->kpt[2], bc);
        I.last_rcenter_rough = det->r_center;
        I.have_rcenter_rough = true;
      }
      I.have_rcenter = true;

      // 走两个后端共用的 PnP（物体点 0.127，见本函数开头的 solve_pose_ippe）
      if (const auto pose = solve_pose_ippe(det->kpt[0], det->kpt[1], det->kpt[2], det->kpt[3])) {
        poses.push_back(*pose);
      }
    }
  } else {
  // 合成自检（--selftest-aim）注入的关键点：两个后端都必须支持。
  // 之前这里只在 sp25 分支处理 ext_kpts，换成 HW26 后端后 self-test 会全部失败。
  if (I.ext_kpts.size() >= 6) {
    const cv::Point2f &pt = I.ext_kpts[0];
    const cv::Point2f &pl = I.ext_kpts[1];
    const cv::Point2f &pb = I.ext_kpts[2];
    const cv::Point2f &pr = I.ext_kpts[3];
    for (const auto &q : I.ext_kpts) I.last_kpt.push_back(q);
    if (const auto pose = solve_pose_ippe(pt, pl, pb, pr)) poses.push_back(*pose);
    DetInfo di;
    di.color = I.cfg.buff_color;
    di.label = 0;
    di.confidence = 1.0f;
    di.cx = pt.x;
    di.cy = pt.y;
    out.dets.push_back(di);
    out.n_det = 1;
    out.n_selected = 1;
    out.max_score = 1.0f;
  } else {
  // 2) letterbox + blob（与 HW26 的 TensorRT 路径完全同构）
  LetterboxInfo info;
  cv::Mat lb = buff_algo::letterbox_image(bgr, info, I.in_size);
  cv::Mat blob =
      cv::dnn::blobFromImage(lb, 1.0 / 255.0, I.in_size, cv::Scalar(), true, false, CV_32F);

  // 3) OpenVINO 推理（替代 TensorRT）
  ov::Tensor in_t(I.compiled.input().get_element_type(), I.compiled.input().get_shape(),
                  blob.ptr<float>());
  I.req.set_input_tensor(in_t);
  I.req.infer();
  const float *raw = I.req.get_output_tensor(0).data<const float>();
  // 全候选最大类别分（不做阈值），用于判断"离检出差多远"
  {
    float mx = 0.0f;
    for (int64_t i = 0; i < kCandidateCount; ++i) {
      const float *row = raw + i * kOutputColumns + 4;
      for (int c = 0; c < 4; ++c) {
        if (row[c] > mx) mx = row[c];
      }
    }
    out.max_score = mx;
  }
  const auto detected =
      buff_algo::decode_detections(raw, kCandidateCount, kOutputColumns, info, I.dcfg);
  out.n_det = static_cast<int>(detected.size());
  // 可视化：把【每一个】检出的关键点都收进来（大符会有 2 组），
  // 而不是只画 detected[0] —— 后者是置信度最高的那个，两片靶标会互相超过，
  // 于是画出来的点在两片之间来回跳。这不影响 tracker 的输入。
  for (const auto &d : detected) {
    std::vector<cv::Point2f> one;
    one.reserve(d.keypoints.size());
    for (const auto &q : d.keypoints) one.emplace_back(q.x(), q.y());
    out.det_kpts.push_back(std::move(one));
  }
  if (!detected.empty()) {
    for (const auto &q : detected[0].keypoints)
      I.last_kpt.push_back(cv::Point2f(q.x(), q.y()));
  }
  out.dets.reserve(detected.size());
  for (const auto &d : detected) {
    DetInfo di;
    di.color = d.detection.color;
    di.label = d.detection.label;
    di.confidence = d.detection.confidence;
    di.cx = d.detection.t.x();
    di.cy = d.detection.t.y();
    out.dets.push_back(di);
  }

  // 4) 目标筛选
  std::vector<BuffDetection> dets;
  dets.reserve(detected.size());
  for (const auto &d : detected) dets.push_back(d.detection);
  std::vector<BuffDetection> selected;
  if (I.cfg.dual_target) {
    // 双靶标：桥自己筛，保留【所有】通过颜色+激活过滤的候选（大符有 2 个）。
    // 不用 HW26 的 select_buffs —— 它结尾写死 dst.push_back(infos[0].buff)，
    // 只保留"离光轴最近"的 1 个。大符两片靶标都在转，谁更近会来回切换 ->
    // 相位每帧跳 72° -> 超过 switch_buff_angle(45.8°) -> BigBuffObserver 把
    // 每一帧都判成"换片"丢弃 -> RANSAC 永远拿不到数据 -> st 卡在 CONVERGING
    // -> ctrl 恒为 0（完全不瞄准）。这就是实测日志里的现象。
    for (const auto &b : dets) {
      if (I.cfg.filter_color && b.color != I.cfg.buff_color) continue;
      if (I.cfg.filter_activation &&
          b.label != static_cast<int>(buff_algo::BuffActivation::INACTIVATE))
        continue;
      selected.push_back(b);
    }
    // 大符最多高亮 2 片，多于 2 个时按置信度取前 2（第 3 个通常是假阳）
    if (selected.size() > 2) {
      std::partial_sort(selected.begin(), selected.begin() + 2, selected.end(),
                        [](const BuffDetection &x, const BuffDetection &y) {
                          return x.confidence > y.confidence;
                        });
      selected.resize(2);
    }
  } else if (I.cfg.filter_color && I.cfg.filter_activation) {
    I.selector->select_buffs(dets, selected);  // HW26 原逻辑（小符：只保留 1 个）
  } else {
    // 放宽过滤（诊断用）：按开启的条件筛选，再取"光轴夹角最小"的一个
    static const std::vector<cv::Point3f> obj_pts = [] {
      std::vector<cv::Point3f> p;
      for (const auto &q : buff_algo::consts::BUFF_POINTS) p.emplace_back(q.x(), q.y(), q.z());
      return p;
    }();
    double best_angle = 1e9;
    int best_idx = -1;
    for (std::size_t i = 0; i < dets.size(); ++i) {
      const auto &b = dets[i];
      if (I.cfg.filter_color && b.color != I.cfg.buff_color) continue;
      if (I.cfg.filter_activation &&
          b.label != static_cast<int>(buff_algo::BuffActivation::INACTIVATE))
        continue;
      const std::vector<cv::Point2f> ip = {
          {b.t.x(), b.t.y()}, {b.l.x(), b.l.y()}, {b.b.x(), b.b.y()}, {b.r.x(), b.r.y()}};
      cv::Mat rvec, tvec;
      try {
        cv::solvePnP(obj_pts, ip, I.cfg.camera_matrix, I.cfg.distort_coeffs, rvec, tvec, false,
                     cv::SOLVEPNP_IPPE);
      } catch (const cv::Exception &) {
        continue;
      }
      const double x = tvec.at<double>(0), y = tvec.at<double>(1), z = tvec.at<double>(2);
      if (std::abs(z) < 1e-6) continue;
      const double ang = std::atan(std::hypot(x, y) / z);
      if (ang < best_angle) {
        best_angle = ang;
        best_idx = static_cast<int>(i);
      }
    }
    // 若"颜色"和"激活"两道闸都关了仍无候选，退化为取置信度最高者
    if (best_idx < 0 && !I.cfg.filter_color && !I.cfg.filter_activation && !dets.empty()) {
      best_idx = 0;
    }
    if (best_idx >= 0) selected.push_back(dets[static_cast<std::size_t>(best_idx)]);
  }
  out.n_selected = static_cast<int>(selected.size());

  // 5) PnP：与 sp25 分支共用同一套（见本函数开头的 solve_pose_ippe，物体点 0.127）。
  //    刻意不调用 I.solver->solve_pnp()：HW26 的 BuffPnPSolver 内部写死
  //    consts::BUFF_POINTS = 0.114，那是它实车靶板的尺寸，用在模拟器上会让
  //    距离偏大约 10%（实测 10.18% vs 0.127 的 0.18%）。
  //    BuffDetection 的 t/l/b/r 与 sp25 的 kpt[0..3] 都是"上/左/下/右"同一批物理点。
  for (const auto &b : selected) {
    if (const auto pose =
            solve_pose_ippe(cv::Point2f(b.t.x(), b.t.y()), cv::Point2f(b.l.x(), b.l.y()),
                            cv::Point2f(b.b.x(), b.b.y()), cv::Point2f(b.r.x(), b.r.y()))) {
      poses.push_back(*pose);
    }
  }
  }  // end if ext_kpts ... else（合成自检 vs 真实推理）
  }  // end if detector==kSp25 ... else

  // 6) 观测系变换 + 质量闸 + 送入 Tracker
  if (!poses.empty()) {
    out.solved = true;
    const Pose3f &p = poses[0];
    const double dis = p.translation.cast<double>().norm();
    out.buff_dis = dis;
    const bool dis_ok = dis >= I.cfg.min_distance && dis <= I.cfg.max_distance;
    out.quality = dis_ok;

    // 6a) 统一先到车体系（云台系 -> 车体系）
    const Eigen::Vector3f t_chassis = R_cg * p.translation;
    const Eigen::Matrix3f Rb_chassis = R_cg * p.rotation.toRotationMatrix();

    // 6b) 从 PnP 位姿取出"符面法向"（扇叶局部 x 轴），并在车体系下平滑。
    //     符本身不动 -> 这个法向在车体系里是常量，所以面系是静止的（不会自激）。
    Eigen::Vector3f n_raw = Rb_chassis.col(0).normalized();
    if (n_raw.dot(t_chassis) > 0.f) n_raw = -n_raw;  // 统一朝相机一侧，避免符号翻转
    if (!I.face_ready) {
      I.n_face = n_raw;
      I.face_ready = true;
    } else {
      const double a = (I.face_frames < I.cfg.face_lock_frames) ? I.cfg.face_ema
                                                                : I.cfg.face_ema_locked;
      I.n_face = ((1.0 - a) * I.n_face + a * n_raw).normalized();
    }

    // 6c) 构造面内基：u0 = 车体 z 轴在面内的投影；u1 = n x u0
    Eigen::Vector3f u0 = Eigen::Vector3f::UnitZ() - I.n_face * I.n_face.z();
    if (u0.norm() < 1e-3f) u0 = Eigen::Vector3f::UnitY() - I.n_face * I.n_face.y();
    u0.normalize();
    const Eigen::Vector3f u1 = I.n_face.cross(u0);
    Eigen::Matrix3f Rc2f;             // 车体系 -> 符面系
    Rc2f.row(0) = I.n_face.transpose();
    Rc2f.row(1) = u0.transpose();
    Rc2f.row(2) = u1.transpose();
    I.R_chassis_face = Rc2f.transpose();  // 符面系 -> 车体系
    ++I.face_frames;

    // 6d) 观测转到所选参考系。抽成 lambda：两个靶标共用同一套变换
    //     （面系由转盘法向决定，与是哪个靶标无关）。
    auto to_obs = [&](const Pose3f &q) -> Pose3f {
      Pose3f o;
      if (I.cfg.frame == RefFrame::kFace) {
        // 强制扇叶局部 z 轴落在面内（其 x 分量置 0），使 e = R*(0,0,-0.7) 的
        // x 分量【精确】为 0 —— 这样 HW26 的固定平面重建才完全精确
        const Eigen::Vector3f t_ch = R_cg * q.translation;
        const Eigen::Matrix3f Rb_ch = R_cg * q.rotation.toRotationMatrix();
        Eigen::Matrix3f Rb_face = Rc2f * Rb_ch;
        Eigen::Vector3f lz = Rb_face.col(2);
        lz.x() = 0.f;
        if (lz.norm() < 1e-3f) lz = Eigen::Vector3f(0.f, 1.f, 0.f);
        lz.normalize();
        const Eigen::Vector3f lx = Eigen::Vector3f::UnitX();
        Eigen::Vector3f ly = lz.cross(lx);
        ly.normalize();
        Eigen::Matrix3f Rfix;
        Rfix.col(0) = lx;
        Rfix.col(1) = ly;
        Rfix.col(2) = lz;
        o.translation = Rc2f * t_ch;
        o.rotation = Eigen::Quaternionf(Rfix).normalized();
      } else if (I.cfg.frame == RefFrame::kChassis) {
        o.translation = R_cg * q.translation;
        o.rotation = Eigen::Quaternionf(R_cg * q.rotation.toRotationMatrix()).normalized();
      } else {  // kGimbal：故意保留以复现自激
        o = q;
      }
      return o;
    };
    Pose3f use = to_obs(p);
    if (I.cfg.frame == RefFrame::kFace) {
      I.c_face_chassis = t_chassis + Rb_chassis * Eigen::Vector3f(0, 0, -0.7f);  // 符心(车体系)
      // 诊断：面系下 e=R*(0,0,-0.7) 的 x 分量（应≈0），以及视角偏离正对的角度
      Eigen::Vector3f e_c = Rb_chassis * Eigen::Vector3f(0, 0, -0.7f);
      out.face_ex = std::abs((Rc2f * e_c).x());
      const Eigen::Vector3f cam_dir = (R_cg.transpose() * Eigen::Vector3f::UnitX());
      out.face_tilt_deg =
          std::acos(std::clamp(std::abs(I.n_face.dot(cam_dir)), 0.f, 1.f)) * 180.0 / M_PI;
    }
    // 6e) 自检：把 PnP 解出的扇叶四角重投影回图像，与检测到的关键点比对。
    //     误差应 < 1px；若很大说明投影链（轴向/内参）有问题。
    if (I.cfg.detector == DetectorKind::kSp25 && I.last_kpt.size() >= 4) {
      const float h = static_cast<float>(I.cfg.blade_half_size);
      const Eigen::Vector3f obj_l[4] = {{0, 0, h}, {0, h, 0}, {0, 0, -h}, {0, -h, 0}};
      const Eigen::Matrix3f Rb_gim = p.rotation.toRotationMatrix();
      double mx = 0.0;
      for (int k = 0; k < 4; ++k) {
        const Eigen::Vector3f pc =
            gimbal_to_cam(p.translation + Rb_gim * obj_l[k]);
        int px = -1, py = -1;
        if (!project_cam(pc, I.cfg.camera_matrix, &px, &py)) continue;
        const double e = std::hypot(static_cast<double>(px) - I.last_kpt[static_cast<std::size_t>(k)].x,
                                    static_cast<double>(py) - I.last_kpt[static_cast<std::size_t>(k)].y);
        mx = std::max(mx, e);
      }
      out.reproj_px = mx;
    }
    // 6f) 双靶标融合（大符）
    //   两片高亮靶标刚性固连在同一个转盘上，所以 roll_B(t) = roll_A(t) + Δ，
    //   Δ 是常数（5 个槽位 -> 72° 的整数倍）。于是同一帧的两个检出是对【同一个】
    //   三角函数体系的两次独立测量，可以融合成一次、噪声降 sqrt(2)；
    //   而且融合后的序列连续，不会再触发 switch_buff_angle 的跳变判据。
    Buff obs(use);
    out.rc_dist_m = static_cast<double>(obs.R_center.norm());
    out.dual_targets = 1;
    out.meas_a_valid = true;
    out.meas_a_ch = R_cg * use.translation;  // A 片实测板心（车体系）
    if (I.cfg.dual_target && poses.size() >= 2) {
      const Pose3f use2 = to_obs(poses[1]);
      const Buff b2(use2);
      const double step = 2.0 * M_PI / 5.0;
      const double d = wrap_pi(static_cast<double>(b2.roll) - static_cast<double>(obs.roll));
      int k = static_cast<int>(std::lround(d / step));
      if (k == 0) k = (d >= 0.0) ? 1 : -1;  // 两靶标不可能同相位
      const double delta = k * step;
      // 诊断：吸附残差与两中心分歧 —— 这两条是"72° 刚性固连"假设的直接检验
      out.dual_resid_deg = std::abs(d - delta) * 180.0 / M_PI;
      out.dual_rcd_m = static_cast<double>((obs.R_center - b2.R_center).norm());
      const double rb_norm = wrap_pi(static_cast<double>(b2.roll) - delta);
      // 圆平均：先把 B 归一化到 A 的相位约定，再取中点
      obs.roll = static_cast<float>(
          wrap_pi(static_cast<double>(obs.roll) + 0.5 * wrap_pi(rb_norm - obs.roll)));
      obs.R_center = 0.5f * (obs.R_center + b2.R_center);  // 转盘中心，两靶标应一致
      out.dual_targets = 2;
      out.dual_delta_deg = delta * 180.0 / M_PI;
      out.meas_b_valid = true;
      out.meas_b_ch = R_cg * use2.translation;  // B 片实测板心（车体系）
    }
    if (dis_ok) I.tracker->push(obs);
  }

  // 7) 状态机推进（无论本帧是否有观测都要 update，与 HW26 example 一致）
  I.tracker->update(t_rel);
  out.status = static_cast<int>(I.tracker->status());
  const BuffState state = I.tracker->get_state();
  out.roll = state.roll;
  out.roll_velocity = state.roll_velocity;

  // 7b) tracker 自己认为的叶片位置（可视化用的"白叉"）。
  //     直接用 HW26 自己的公式与半径 consts::BUFF_RADIUS，保证和它内部一致 ——
  //     这样"白叉落在实测量上"才能证明跟踪锁的是真靶标。
  {
    const float R = buff_algo::consts::BUFF_RADIUS;
    const Eigen::Vector3f p_obs(state.r_center.x(),
                                state.r_center.y() - R * std::cos(state.roll),
                                state.r_center.z() + R * std::sin(state.roll));
    // 观测系 -> 车体系（tracker 的状态是在 feed 它时所用的参考系里）
    auto obs2ch = [&](const Eigen::Vector3f &q) -> Eigen::Vector3f {
      if (I.cfg.frame == RefFrame::kFace) return I.R_chassis_face * q;
      if (I.cfg.frame == RefFrame::kChassis) return q;
      return R_cg * q;
    };
    out.trk_rc_ch = obs2ch(Eigen::Vector3f(state.r_center));
    out.trk_u_ch = obs2ch(Eigen::Vector3f(0.f, -1.f, 0.f));  // 轨道面内基（与公式一致）
    out.trk_v_ch = obs2ch(Eigen::Vector3f(0.f, 0.f, 1.f));
    out.trk_blade_ch = obs2ch(p_obs);
    out.trk_valid = true;
  }

  // 8) 预测 + 弹道 + 云台角
  const bool trackable = (out.status == static_cast<int>(StatusType::TRACKING));
  double lead = 0.0, fly = 0.0, pitch_need = 0.0;
  Eigen::Vector3f aim_pt = Eigen::Vector3f::Zero();
  bool aim_valid = false;

  if (out.solved && trackable) {
    const double alg_lag =
        (I.cfg.alg_latency > 0.0) ? I.cfg.alg_latency : I.last_latency_ms / 1000.0;
    const double base_lead = alg_lag + I.cfg.predict_time + I.cfg.gimbal_lead_time;

    // 关键：必须先把状态喂给 predictor，否则 predict_position 恒返回 (0,0,0)
    I.predictor->set_state(state, t_rel, t_rel);

    // 观测系 -> 车体系（弹道必须在重力对齐、且以枪口为原点的坐标系里解）
    auto to_chassis = [&](const Eigen::Vector3f &p_obs) -> Eigen::Vector3f {
      Eigen::Vector3f p_ch;
      if (I.cfg.frame == RefFrame::kFace) {
        p_ch = I.R_chassis_face * p_obs;              // 符面系 -> 车体系
      } else if (I.cfg.frame == RefFrame::kChassis) {
        p_ch = p_obs;
      } else {
        p_ch = R_cg * p_obs;                          // 云台系 -> 车体系
      }
      if (I.cfg.use_muzzle_offset) p_ch -= (R_cg * t_muzzle);  // 枪口平移（云台系）
      return p_ch;
    };

    lead = base_lead;
    for (int it = 0; it < 5; ++it) {
      aim_pt = I.predictor->predict_position(static_cast<float>(lead));
      const Eigen::Vector3f p_ch = to_chassis(aim_pt);
      const double d = std::hypot(static_cast<double>(p_ch.x()), static_cast<double>(p_ch.y()));
      const double hgt = static_cast<double>(p_ch.z());
      if (!solve_ballistic_no_drag(d, hgt, I.cfg.bullet_speed, I.cfg.gravity, &pitch_need, &fly)) {
        aim_valid = false;
        break;
      }
      const double new_lead = base_lead + fly;
      aim_valid = true;
      if (std::abs(new_lead - lead) < 1e-3) {
        lead = new_lead;
        break;
      }
      lead = new_lead;
    }

    if (aim_valid) {
      const Eigen::Vector3f p_ch = to_chassis(aim_pt);
      const double yaw_cmd =
          std::atan2(static_cast<double>(p_ch.y()), static_cast<double>(p_ch.x())) +
          I.cfg.yaw_offset_deg * M_PI / 180.0;
      const double pitch_offset_rad = I.cfg.pitch_offset_deg * M_PI / 180.0;
      double pitch_cmd = -(pitch_need + pitch_offset_rad);
      if (pitch_cmd > kSimPitchLimit) pitch_cmd = kSimPitchLimit;
      if (pitch_cmd < -kSimPitchLimit) pitch_cmd = -kSimPitchLimit;

      // ---- 先算出"未来打击点"的绝对位置与像素（绿点用 raw，不受护栏影响）----
      if (I.cfg.frame == RefFrame::kFace) {
        I.aim_ch_abs = I.R_chassis_face * aim_pt;
      } else if (I.cfg.frame == RefFrame::kChassis) {
        I.aim_ch_abs = aim_pt;
      } else {
        I.aim_ch_abs = R_cg * aim_pt;
      }
      I.aim_ch_valid = true;
      {
        const Eigen::Vector3f p_gim = R_cg.transpose() * I.aim_ch_abs;
        project_cam(gimbal_to_cam(p_gim), I.cfg.camera_matrix, &out.aim_px_x, &out.aim_px_y);
      }

      // ================= ④ sp25 桥的四道护栏 =================
      // (1) aim 像素瞬移拒帧：PnP 翻转 / 换扇叶会让 aim 点瞬移 -> 本帧不伺服
      bool aim_jumped = false;
      if (I.have_prev_aim_px && out.aim_px_x > 0) {
        const double step = std::hypot(static_cast<double>(out.aim_px_x) - I.prev_aim_px.x,
                                       static_cast<double>(out.aim_px_y) - I.prev_aim_px.y);
        aim_jumped = step > I.cfg.max_aim_step_px;
        out.aim_jumped = aim_jumped;
      }
      if (!aim_jumped) {
        // (2) 角速度限幅：clamp(raw - 当前姿态, ±slew*dt)（与帧率无关）
        double dt = 0.033;
        if (I.prev_t_rel > 0 && t_rel > I.prev_t_rel) dt = t_rel - I.prev_t_rel;
        dt = std::clamp(dt, 0.001, 0.1);
        const double slew = I.cfg.max_slew_deg_s * M_PI / 180.0;
        const double dyaw = std::clamp(wrap_pi(yaw_cmd - gimbal_yaw), -slew * dt, slew * dt);
        const double dpit = std::clamp(pitch_cmd - gimbal_pitch, -slew * dt, slew * dt);
        double yaw_out = wrap_pi(gimbal_yaw + dyaw);
        double pitch_out = gimbal_pitch + dpit;
        // (3) 低通：cmd = (1-a)*prev + a*raw
        if (I.have_prev_cmd_out) {
          const double a = std::clamp(I.cfg.cmd_smooth_alpha, 0.01, 1.0);
          yaw_out = wrap_pi(I.prev_cmd_out_yaw + a * wrap_pi(yaw_out - I.prev_cmd_out_yaw));
          pitch_out = I.prev_cmd_out_pitch + a * (pitch_out - I.prev_cmd_out_pitch);
        }
        // (4) 记录历史（control=false 时会被清空）
        I.prev_cmd_out_yaw = yaw_out;
        I.prev_cmd_out_pitch = pitch_out;
        I.have_prev_cmd_out = true;
        I.prev_aim_px = cv::Point2f(static_cast<float>(out.aim_px_x),
                                    static_cast<float>(out.aim_px_y));
        I.have_prev_aim_px = true;

        out.yaw = yaw_out;
        out.pitch = pitch_out;
        out.control = true;
      }
      // 开火闸用【原始弹道解】对当前姿态（与 ④ 一致：指令发出≠云台到位）
      out.fire_aim_yaw = yaw_cmd;
      out.fire_aim_pitch = pitch_cmd;
      I.prev_t_rel = t_rel;
      if (I.have_prev_aim) {
        const double dy = wrap_pi(yaw_cmd - I.prev_cmd_yaw);
        const double dp = pitch_cmd - I.prev_cmd_pitch;
        out.cmd_jump_deg = std::hypot(dy, dp) * 180.0 / M_PI;
      }
      I.prev_cmd_yaw = yaw_cmd;
      I.prev_cmd_pitch = pitch_cmd;
      I.have_prev_aim = true;




      // 9) 开火判定（用原始弹道解 vs 当前云台姿态）
      const double fire_dyaw = wrap_pi(yaw_cmd - gimbal_yaw);
      const double fire_dpitch = pitch_cmd - gimbal_pitch;
      out.fire_err = std::hypot(fire_dyaw, fire_dpitch);
      if (out.fire_err <= I.cfg.fire_err_rad) {
        ++I.align_run;
      } else {
        I.align_run = 0;
      }
      if (I.cfg.fire_enabled && I.align_run >= I.cfg.fire_stable_frames &&
          (t_rel - I.last_fire_t) >= I.cfg.fire_cooldown) {
        out.fire = true;
        I.align_run = 0;
        I.last_fire_t = t_rel;
      }
    }
  } else {
    I.align_run = 0;
    // ④ 同款：控制无效帧清空平滑历史，恢复时从当前云台姿态重新起跳，不叠加旧命令
    I.have_prev_cmd_out = false;
    I.have_prev_aim_px = false;
  }

  // 9b) 两个可视化点（总是计算；仅 visualize 时绘制）
  {
    auto project_px = [&](const Eigen::Vector3f &p_ch, int *px, int *py) -> bool {
      const Eigen::Vector3f p_gim = R_cg.transpose() * p_ch;
      return project_cam(gimbal_to_cam(p_gim), I.cfg.camera_matrix, px, py);
    };
    if (I.aim_ch_valid) project_px(I.aim_ch_abs, &out.green_px_x, &out.green_px_y);
    if (I.face_ready) {
      // 枪管相对云台固定上仰 20deg；云台物理俯仰 = -gimbal_pitch（模拟器 pitch<0 抬枪）
      const double barrel_elev =
          -static_cast<double>(gimbal_pitch) + (-I.cfg.pitch_offset_deg) * M_PI / 180.0;
      const double gyaw = static_cast<double>(gimbal_yaw);
      const Eigen::Vector3f dir_ch(static_cast<float>(std::cos(barrel_elev) * std::cos(gyaw)),
                                   static_cast<float>(std::cos(barrel_elev) * std::sin(gyaw)),
                                   static_cast<float>(std::sin(barrel_elev)));
      const Eigen::Vector3f muzzle_ch = R_cg * t_muzzle;
      const float denom = I.n_face.dot(dir_ch);
      if (std::abs(denom) > 1e-4f) {
        const float tt = I.n_face.dot(I.c_face_chassis - muzzle_ch) / denom;
        if (tt > 0.f && tt < 60.f) {
          out.blue_range = tt;
          if (project_px(muzzle_ch + tt * dir_ch, &out.blue_px_x, &out.blue_px_y)) {
            out.blue_valid = true;
          }
        }
      }
    }
  }

  if (I.cfg.visualize) {
    // 车体系 -> 像素（可视化块自己的助手；上面 aim 段那个 project_px 不在此作用域）
    auto proj_ch = [&](const Eigen::Vector3f &p_ch, int *px, int *py) -> bool {
      return project_cam(gimbal_to_cam(R_cg.transpose() * p_ch), I.cfg.camera_matrix, px, py);
    };
    out.debug = bgr.clone();
    const int cx = static_cast<int>(std::lround(I.cfg.camera_matrix.at<double>(0, 2)));
    const int cy = static_cast<int>(std::lround(I.cfg.camera_matrix.at<double>(1, 2)));
    cv::drawMarker(out.debug, {cx, cy}, cv::Scalar(128, 128, 128), cv::MARKER_CROSS, 20, 2);

    // ---- 绿点：未来打击点（预测命中点）----
    if (out.green_px_x > 0) {
      cv::circle(out.debug, {out.green_px_x, out.green_px_y}, 8, cv::Scalar(0, 255, 0),
                 cv::FILLED, cv::LINE_AA);
    }
    // ---- 蓝点：当前枪管射线 ∩ 整个符所在平面（直径与绿点相同）----
    if (out.blue_valid) {
      cv::circle(out.debug, {out.blue_px_x, out.blue_px_y}, 8, cv::Scalar(255, 0, 0),
                 cv::FILLED, cv::LINE_AA);
    }

    // ---- 把检测到的东西画出来（让"在检测什么"可见）----
    // 多个检出各自一组颜色：第 1 组黄（与原行为一致），第 2 组品红，第 3 组起循环。
    // 以前只画 detected[0]，大符时会在两片靶标之间来回跳。
    {
      static const cv::Scalar kGroupColor[3] = {
          cv::Scalar(0, 255, 255),   // 黄
          cv::Scalar(255, 0, 255),   // 品红
          cv::Scalar(0, 128, 255),   // 橙
      };
      for (std::size_t g = 0; g < out.det_kpts.size(); ++g) {
        const cv::Scalar &col = kGroupColor[g < 3 ? g : 2];
        const bool is_second = (g == 1);
        for (const auto &d : out.det_kpts[g]) {
          cv::circle(out.debug, {static_cast<int>(d.x), static_cast<int>(d.y)},
                     is_second ? 3 : 4, col, cv::FILLED, cv::LINE_AA);
        }
        // 在每组第一个点旁标出组号，便于区分是哪一片靶标
        if (!out.det_kpts[g].empty()) {
          const cv::Point2f &p0 = out.det_kpts[g].front();
          char tag[8];
          std::snprintf(tag, sizeof(tag), "%zu", g + 1);
          cv::putText(out.debug, tag, {static_cast<int>(p0.x) + 6, static_cast<int>(p0.y) - 6},
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv::LINE_AA);
        }
      }
    }
    if (I.have_rcenter) {
      cv::circle(out.debug, {static_cast<int>(I.last_rcenter.x), static_cast<int>(I.last_rcenter.y)},
                 10, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);  // 中心 R（青）
    }

    // ---- 大符判据：tracker 认为的叶片（白叉）该落在哪一片实测靶标上？----
    //   白叉 = tracker 的唯一相位状态推出的叶片位置。它落在【亮绿或暗绿实测点】上
    //   -> 跟踪锁的是真靶标 ✓；落在两点【正中间】-> 相位被平均歪了 ✗；
    //   落在圆外/别处 -> 跟踪错了 ✗。
    if (out.trk_valid && I.face_ready) {
      // 轨道圆（半径 = consts::BUFF_RADIUS），48 段折线
      const float R = buff_algo::consts::BUFF_RADIUS;
      cv::Point prev(-1, -1);
      for (int i = 0; i <= 48; ++i) {
        const double th = 2.0 * M_PI * i / 48.0;
        const Eigen::Vector3f p_ch = out.trk_rc_ch + R * (std::cos(th) * out.trk_u_ch +
                                                          std::sin(th) * out.trk_v_ch);
        int px = -1, py = -1;
        cv::Point cur(-1, -1);
        if (proj_ch(p_ch, &px, &py)) cur = cv::Point(px, py);
        if (prev.x >= 0 && cur.x >= 0) {
          cv::line(out.debug, prev, cur, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);
        }
        prev = cur;
      }
      // 两片各自实测的板心
      int ax = -1, ay = -1, bx = -1, by = -1;
      const bool okA = out.meas_a_valid && proj_ch(out.meas_a_ch, &ax, &ay);
      const bool okB = out.meas_b_valid && proj_ch(out.meas_b_ch, &bx, &by);
      if (okA) cv::circle(out.debug, {ax, ay}, 6, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
      if (okB) cv::circle(out.debug, {bx, by}, 6, cv::Scalar(0, 160, 0), cv::FILLED, cv::LINE_AA);
      // tracker 认为的叶片位置：白叉
      int tx = -1, ty = -1;
      if (proj_ch(out.trk_blade_ch, &tx, &ty)) {
        const cv::Scalar w(255, 255, 255);
        cv::line(out.debug, {tx - 9, ty - 9}, {tx + 9, ty + 9}, w, 2, cv::LINE_AA);
        cv::line(out.debug, {tx - 9, ty + 9}, {tx + 9, ty - 9}, w, 2, cv::LINE_AA);
      }
    }
    char txt[256];
    const char *st = (out.status == 0)   ? "CONVERGING"
                     : (out.status == 1) ? "TRACKING"
                     : (out.status == 2) ? "TEMP_LOST"
                                         : "LOST";
    std::snprintf(txt, sizeof(txt), "%s det=%d sel=%d dis=%.2f roll=%.1f w=%.2f err=%.4f%s", st,
                  out.n_det, out.n_selected, out.buff_dis, out.roll * 180 / M_PI,
                  out.roll_velocity, out.fire_err, out.fire ? "  FIRE" : "");
    cv::putText(out.debug, txt, {12, 34}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                out.fire ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    std::snprintf(txt, sizeof(txt), "lat=%.0fms lead=%.3fs fly=%.3fs", out.latency_ms, lead, fly);
    cv::putText(out.debug, txt, {12, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    std::snprintf(txt, sizeof(txt), "GREEN=future hit  BLUE=barrel-on-rune-plane(%.2fm)  CYAN=R center",
                  out.blue_range);
    cv::putText(out.debug, txt, {12, 88}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    // 双靶标自检：nt=融合了几片  dlt=吸附后的 Δ  dres=吸附残差(应为几度)
    //             rcd=两片解出的转盘中心分歧(应为几 cm)  rc=到转盘中心距离(标定用)
    std::snprintf(txt, sizeof(txt),
                  "nt=%d dlt=%.0f dres=%.1fdeg rcd=%.2fm rc=%.2fm  "
                  "白叉=tracker认为的叶片(该落在绿点上)",
                  out.dual_targets, out.dual_delta_deg, out.dual_resid_deg, out.dual_rcd_m,
                  out.rc_dist_m);
    cv::putText(out.debug, txt, {12, 112}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                (out.dual_targets == 2) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 140, 255), 1,
                cv::LINE_AA);
  }

  const auto t_end = std::chrono::steady_clock::now();
  out.lead_s = lead;
  out.fly_s = fly;
  out.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  I.last_latency_ms = out.latency_ms;

  // 10) 日志（含 ④ 日志里最有用的那些量，避免重犯"无日志"的错）
  if (I.log) {
    const bool interesting = out.fire || (I.frame_count % std::max(1, I.cfg.log_every) == 0);
    if (interesting) {
      // Absolute image timestamp (unix-epoch ns) so this CSV can be joined with
      // any other log on the same clock. The old column here was t_rel, a
      // RELATIVE time since the bridge anchored, which cannot be joined.
      I.log << I.frame_count << ',' << sim_timestamp_ns << ',' << out.latency_ms << ','
            << gimbal_yaw * 180.0 / M_PI << ',' << gimbal_pitch * 180.0 / M_PI << ',' << out.n_det
            << ',' << out.n_selected << ',' << (out.solved ? 1 : 0) << ','
            << (out.quality ? 1 : 0) << ',' << out.status << ',' << out.buff_dis << ',' << out.roll
            << ',' << out.roll_velocity << ',' << lead << ',' << fly << ',' << out.aim_px_x << ','
            << out.aim_px_y << ',' << out.yaw * 180.0 / M_PI << ','
            << out.pitch * 180.0 / M_PI << ',' << out.fire_err << ',' << (out.control ? 1 : 0)
            << ',' << (out.fire ? 1 : 0) << ','
            << out.face_ex << ',' << out.reproj_px << ','
            << gimbal_roll * 180.0 / M_PI << ',' << out.face_tilt_deg << '\n';
      I.log.flush();
    }
  }
  ++I.frame_count;
  (void)dt_frame;
  return out;
}

}  // namespace hw26_bridge
