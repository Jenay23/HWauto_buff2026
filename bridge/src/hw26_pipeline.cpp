#include "hw26_pipeline.hpp"

#include "local_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "buff_algo/common/types.hpp"

namespace {
struct StageAcc {
  static constexpr int kN = 4;
  const char *names[kN] = {"letterbox", "infer", "decode", "draw"};
  double acc[kN] = {0,0,0,0}; long cnt[kN] = {0,0,0,0}; long frames = 0;
  std::chrono::steady_clock::time_point t;
  void mark(int i) {
    auto now = std::chrono::steady_clock::now();
    if (i > 0) { acc[i-1] += std::chrono::duration<double,std::milli>(now-t).count(); ++cnt[i-1]; }
    t = now;
  }
  void report(double lat) {
    ++frames; if (frames % 100 != 0) return;
    double sum = 0;
    for (int i=0;i<kN;++i) sum += cnt[i] ? acc[i]/cnt[i] : 0.0;
    std::printf("[stage] 每帧平均(ms):");
    for (int i=0;i<kN;++i) std::printf(" %s=%.2f", names[i], cnt[i]?acc[i]/cnt[i]:0.0);
    std::printf(" 其余=%.2f 合计=%.2f (lat=%.2f)\n", lat-sum, sum, lat);
    std::fflush(stdout);
    for (int i=0;i<kN;++i) { acc[i]=0; cnt[i]=0; }
  }
};
StageAcc g_stage_acc;
}

namespace {
// 设备编译的兜底：GPU/NPU 编译失败时自动退回 CPU，避免桥直接崩掉。
template <typename T>
inline ov::CompiledModel compile_with_fallback(ov::Core &core, const T &thing,
                                               const std::string &dev) {
  if (dev == "CPU") return core.compile_model(thing, dev);
  try {
    return core.compile_model(thing, dev);
  } catch (const std::exception &e) {
    std::cerr << "[hw26] 警告: 设备 " << dev << " 编译失败，自动回退 CPU: " << e.what() << std::endl;
    return core.compile_model(thing, "CPU");
  }
}
}  // namespace

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

constexpr int64_t kCandidateCount = 5040;
constexpr int64_t kOutputColumns = 26;  // 4 + 4 + 9*2
// 用它把"配置里只写一个 model: 路径"变成可能：后端由输出形状判定。
constexpr double kSimPitchLimit = 0.785;

double wrap_pi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

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
      # 实测：一次点亮窗口只有约 58 帧（亮 ~58 / 灭 ~55），而 det=0 会让 tracker
      # 进 LOST、下次进 CONVERGING 时 initialize() 把 RANSAC 数据全清掉 -> 无法跨
      # 窗口累积。所以 100 永远够不到（58 < 100），st 恒为 0。降到 35 让一个窗口内
      # 可以 ready。【待验证】期望 st 能到 1(TRACKING)。
      min_inliers: 35
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

// ============================================================================
// 固定半径圆的圆心拟合（center_source=1 用）
// ============================================================================
// 只拿【一片】靶标的历史位置 t（车体系）来解转盘圆心 C。
//
// 原理：|t - C| = R 是硬约束（R = 0.7 m，即 consts::BUFF_RADIUS）。展开：
//     |t|² - 2 t·C + |C|² = R²  ->  2 t_y·C_y + 2 t_z·C_z - k = |t|²
// 其中 k = |C|² - R²。于是对 (C_y, C_z, k) 是**线性**最小二乘 —— 不需要迭代、
// 不需要初值；而且样本 ≥3（实际要求更多）后就没有"中垂线两支"的二义性：
// 两支里只有一支能同时满足所有样本。
//
// 为什么不用 PnP 朝向反推：四个物点 (0,0,±h),(0,±h,0) 构成正方形，绕法向每转
// 90° 自重合 -> 面内朝向有 4 重退化，实测两片由此解出的圆心分歧 0.8~1.6 m。
// 位置 t 没有这个退化。
//
// 外点剔除：该片自己发生 IPPE 双解翻转的那几帧，t 会明显偏离圆，必须稳住。
// 用 **IRLS（迭代重加权最小二乘，Cauchy 权）** 而不是"一次最小二乘 + 硬残差门"：
// 后者会被外点整体拽偏，于是连正常点的残差都超门，结果把好点也一起剔掉
// （实测 40 点里 5 个 0.3 m 的外点就只剩 14 个、圆心偏 0.4 m）。
// 也不用 RANSAC：本模型的三点最小集会把"半径 0.7"当成自由量（3 未知量 3 方程），
// 假设半径不受约束、噪声放大严重，抽出来的圆心本身就飘。
// IRLS 保留了线性结构，靠尺度自适应的权重把外点压下去，不需要任何初值。
bool fit_fixed_radius_center(
    const std::deque<std::pair<double, Eigen::Vector3f>> &hist, const Eigen::Matrix3f &Rc2f,
    double radius, int min_n, double min_span_m, double gate_m, double t_now, double keep_s,
    Eigen::Vector3f *c_out, int *n_out, double *rms_out) {
  // 投影到【当前】面系。历史存的是车体系坐标，所以面系法向的缓慢更新会自动
  // 作用到整段历史上（不会因为 n 漂移而把历史算歪）。
  std::vector<Eigen::Vector3f> p;
  p.reserve(hist.size());
  for (const auto &h : hist) {
    if (keep_s > 0.0 && (t_now - h.first) > keep_s) continue;
    p.push_back(Rc2f * h.second);
  }
  const int n = static_cast<int>(p.size());
  if (n < min_n) return false;

  // 弧太短 -> 在"垂直于弦"的方向上病态，解出来的圆心会飞。先查跨度。
  double span = 0.0;
  for (int i = 0; i < n; ++i) span = std::max(span, static_cast<double>((p[i] - p[0]).norm()));
  if (span < min_span_m) return false;

  auto resid = [&](int i, double cy, double cz) {
    return std::abs(std::hypot(p[i].y() - cy, p[i].z() - cz) - radius);
  };
  // 加权线性最小二乘（3 个未知量：(C_y, C_z, k)）；w 为空表示等权
  auto wls = [&](const std::vector<double> &w, double *cy, double *cz, double *rms,
                 const std::vector<int> *subset) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    double wsum = 0.0;
    for (int i = 0; i < n; ++i) {
      if (subset && std::find(subset->begin(), subset->end(), i) == subset->end()) continue;
      const double wi = w.empty() ? 1.0 : w[i];
      if (wi <= 0.0) continue;
      const double y = p[i].y(), z = p[i].z();
      const Eigen::Vector3d phi(2.0 * y, 2.0 * z, -1.0);
      A += wi * phi * phi.transpose();
      b += wi * phi * (y * y + z * z);
      wsum += wi;
    }
    if (wsum <= 0.0) return false;
    A.diagonal().array() += 1e-9;  // 轻微 ridge：样本几乎共线时也不炸
    const Eigen::Vector3d s = A.ldlt().solve(b);
    if (!s.allFinite()) return false;
    *cy = s(0);
    *cz = s(1);
    double se = 0.0;
    double cnt = 0.0;
    for (int i = 0; i < n; ++i) {
      if (subset && std::find(subset->begin(), subset->end(), i) == subset->end()) continue;
      const double d = resid(i, *cy, *cz);
      se += d * d;
      cnt += 1.0;
    }
    *rms = std::sqrt(se / std::max(1.0, cnt));
    return true;
  };

  // ---- 初始等权最小二乘 ----
  double cy = 0.0, cz = 0.0, rms = 0.0;
  if (!wls({}, &cy, &cz, &rms, nullptr)) return false;

  // ---- 鲁棒 Gauss-Newton：直接对【真实残差】|t-C| - R 做 Cauchy 加权 ----
  // 注意不能拿线性化模型的残差当权重：那套参数化把半径当自由量，权重与真实
  // 残差不一致，实测压不住外点。这里 2 个未知量 (Cy,Cz)，每个点一行雅可比
  //   ∂(|t-C|-R)/∂C = -(t-C)/|t-C| = -u
  // 正规方程 H dC = Σ w·u·r，2×2 直接解。
  auto gn_step = [&](const std::vector<double> *wsub, double damp) -> bool {
    double h00 = 0.0, h01 = 0.0, h11 = 0.0, g0 = 0.0, g1 = 0.0;
    for (int i = 0; i < n; ++i) {
      if (wsub && (*wsub)[i] <= 0.0) continue;
      const double dy = p[i].y() - cy, dz = p[i].z() - cz;
      const double d = std::max(1e-6, std::hypot(dy, dz));
      const double uy = dy / d, uz = dz / d;
      const double wi = wsub ? (*wsub)[i] : 1.0;
      const double r = d - radius;
      h00 += wi * uy * uy;
      h01 += wi * uy * uz;
      h11 += wi * uz * uz;
      g0 += wi * uy * r;
      g1 += wi * uz * r;
    }
    h00 += damp;
    h11 += damp;
    const double det = h00 * h11 - h01 * h01;
    if (std::abs(det) < 1e-12) return false;
    const double dcy = (h11 * g0 - h01 * g1) / det;
    const double dcz = (-h01 * g0 + h00 * g1) / det;
    const double step = std::hypot(dcy, dcz);
    const double kMaxStep = 0.3;
    const double sc = (step > kMaxStep) ? kMaxStep / step : 1.0;
    cy += sc * dcy;
    cz += sc * dcz;
    return sc * step > 1e-7;
  };

  // 迭代：尺度 s = 1.4826·median(|r|)，Cauchy 权 w = 1/(1+(r/(3s))²)，再做一步 GN
  std::vector<double> w(n, 1.0);
  for (int it = 0; it < 40; ++it) {
    std::vector<double> absr(n, 0.0);
    for (int i = 0; i < n; ++i) absr[i] = resid(i, cy, cz);
    std::vector<double> sorted = absr;
    std::sort(sorted.begin(), sorted.end());
    const double s = std::max(1e-3, 1.4826 * sorted[sorted.size() / 2]);
    for (int i = 0; i < n; ++i) {
      const double u = absr[i] / (3.0 * s);
      w[i] = 1.0 / (1.0 + u * u);
    }
    if (!gn_step(&w, 1e-9)) break;
  }

  // ---- 鲁棒 Gauss-Newton：直接对【真实残差】|t-C| - R 做 Cauchy 加权 ----
  // ---- 定内点集（用于报 n/rms），并在内点上做一次无权的 GN 收尾 ----
  std::vector<double> absr(n, 0.0);
  for (int i = 0; i < n; ++i) absr[i] = resid(i, cy, cz);
  std::vector<double> sorted = absr;
  std::sort(sorted.begin(), sorted.end());
  const double s_final = std::max(1e-3, 1.4826 * sorted[sorted.size() / 2]);
  const double inl_gate = std::max(gate_m > 0.0 ? gate_m : 0.05, 3.0 * s_final);
  std::vector<int> idx;
  idx.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (absr[i] <= inl_gate) idx.push_back(i);
  }
  if (static_cast<int>(idx.size()) < min_n) return false;
  std::vector<double> w_in(n, 0.0);
  for (int i : idx) w_in[i] = 1.0;
  for (int it = 0; it < 10; ++it) {
    if (!gn_step(&w_in, 1e-9)) break;
  }
  rms = 0.0;
  for (int i : idx) rms += resid(i, cy, cz) * resid(i, cy, cz);
  rms = std::sqrt(rms / static_cast<double>(idx.size()));

  // x（出平面）取参与拟合样本的均值 —— 同一转盘的两片应当共面，这一项应≈常数。
  double xs = 0.0;
  for (int i : idx) xs += p[i].x();
  xs /= static_cast<double>(idx.size());
  *c_out =
      Eigen::Vector3f(static_cast<float>(xs), static_cast<float>(cy), static_cast<float>(cz));
  if (n_out) *n_out = static_cast<int>(idx.size());
  if (rms_out) *rms_out = rms;
  return true;
}

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
  if (out_rows == kCandidateCount && out_cols == kOutputColumns) {
    I.cfg.detector = DetectorKind::kHw26;
    I.cfg.blade_half_size = I.cfg.blade_half_size_hw26;
  } else {
    std::ostringstream os;
    os << "[hw26] FATAL: model output shape is not HW26 [1," << kCandidateCount << ","
       << kOutputColumns << "] , got [";
    for (size_t i = 0; i < oshp.size(); ++i) os << (i ? "," : "") << oshp[i];
    os << "]. Check the model: line in bridge.yaml.";
    throw std::runtime_error(os.str());
  }

  I.compiled = compile_with_fallback(I.core, model, cfg.device);
  { std::cout << "[hw26] OpenVINO device=" << cfg.device << "  available:";
    for (const auto &d : I.core.get_available_devices()) std::cout << " " << d;
    std::cout << std::endl; }
  I.req = I.compiled.create_infer_request();
  std::cout << "[hw26] detector=hw26(" << cfg.model_path
            << ")  (9 keypoints, colour + activation classified)" << std::endl;

  I.selector = std::make_unique<buff_algo::BuffSelector>(cfg.buff_color);
  I.selector->set_camera_matrix(cfg.camera_matrix, cfg.distort_coeffs);
  I.solver = std::make_unique<buff_algo::BuffPnPSolver>();
  I.solver->set_camera_matrix(cfg.camera_matrix, cfg.distort_coeffs);

  if (!I.tracker_fs.open(kDefaultTrackerYaml, cv::FileStorage::READ | cv::FileStorage::MEMORY)) {
    throw std::runtime_error("hw26: failed to open embedded tracker yaml");
  }
  I.tracker = std::make_unique<buff_algo::BuffTracker>(I.tracker_fs["buff_tracker"]);
  I.tracker->set_mode(cfg.buff_mode == 2 ? BuffMode::BIG_BUFF : BuffMode::SMALL_BUFF);
  // ★ 关键：local_model 必须【真正】作用到 local_trk，否则它一直用默认 mode 0
  I.local_trk.set_mode(cfg.local_model);
  I.local_trk.set_rms_gates(cfg.local_rms_gate_deg);
  I.local_trk.set_coast_s(cfg.local_coast_s);
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
               //   local_theta_rad 实际喂进局部 tracker 的相位（弦或圆心路径）
               //   local_chord     1 = 该相位来自"弦相位"（圆心无关）
               "ex_m,reproj_px,gimbal_roll_deg,tilt_deg,local_theta_rad,local_chord\n";
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
  g_stage_acc.mark(0);
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
  // 与 poses 平行的【来源下标】（对应 out.det_kpts / detected 的顺序）。
  // 可视化要把黄/品红的组号稳定地绑在同一片靶标上，就需要这个映射；
  // 融合块在更外层，所以必须声明在这里。
  std::vector<std::size_t> selected_src;
  // 与 poses 平行的来源下标。**组号必须用这个**（不能用与 selected 平行的
  // selected_src）：某片 PnP 失败时 poses 会变短，两者一错位，黄/品红就会
  // 落到错误的靶标上 —— 这正是"亮绿/暗绿稳、黄/品红跳"的原因。
  std::vector<std::size_t> pose_src;
  std::vector<float> cur_rolls;   // 本帧两片各自的 roll（诊断用）
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
                             const cv::Point2f &pb, const cv::Point2f &pr,
                             int *pick_out = nullptr,
                             std::vector<Pose3f> *all_out = nullptr) -> std::optional<Pose3f> {
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

    // 把 IPPE 的【全部解】也导出（云台系），供"两片联合消歧"使用：
    // 单片连续性判据在遮挡/翻转时不可靠，而两片同属一个刚性转盘，几何约束强得多。
    if (all_out) {
      all_out->clear();
      for (std::size_t k = 0; k < rvecs.size(); ++k) {
        const Eigen::Vector3f t_c(static_cast<float>(tvecs[k].at<double>(0)),
                                  static_cast<float>(tvecs[k].at<double>(1)),
                                  static_cast<float>(tvecs[k].at<double>(2)));
        cv::Mat Rk;
        cv::Rodrigues(rvecs[k], Rk);
        Eigen::Matrix3f Re;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) Re(a, b) = static_cast<float>(Rk.at<double>(a, b));
        Pose3f q;
        q.translation = R_cam2gim * t_c;
        q.rotation = Eigen::Quaternionf(R_cam2gim * Re).normalized();
        all_out->push_back(q);
      }
    }

    // 双解消歧：首帧取重投影误差小者；此后取与上一帧更连续者
    int pick = 0;
    if (rvecs.size() == 1) {
      pick = 0;
    } else if (I.prev_ts.empty()) {
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
        // 与上一帧【每一个】位姿都算一遍代价，取最小 —— 不依赖"这片是 1 号还是 2 号"
        double dt = 1e18;
        double dq = 1e18;
        for (std::size_t h = 0; h < I.prev_ts.size(); ++h) {
          const double dth = (t_g - I.prev_ts[h]).norm();
          const double dqh = 2.0 * std::acos(std::min(
              1.0, std::abs(static_cast<double>(q.dot(I.prev_qs[h])))));
          if (2.0 * dth + dqh < 2.0 * dt + dq) {
            dt = dth;
            dq = dqh;
          }
        }
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
    out.n_solutions = static_cast<int>(rvecs.size());
    out.sol_pick = pick;
    if (pick_out) *pick_out = pick;
    return p;
  };

  {
  // 合成自检（--selftest-aim）注入的关键点。
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
    g_stage_acc.mark(1);

  // 3) OpenVINO 推理（替代 TensorRT）
  ov::Tensor in_t(I.compiled.input().get_element_type(), I.compiled.input().get_shape(),
                  blob.ptr<float>());
  I.req.set_input_tensor(in_t);
  I.req.infer();
    g_stage_acc.mark(2);
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
    g_stage_acc.mark(3);
  out.n_det = static_cast<int>(detected.size());
  // 可视化：把【每一个】检出的关键点都收进来（大符会有 2 组），
  // 而不是只画 detected[0] —— 后者是置信度最高的那个，两片靶标会互相超过，
  // 于是画出来的点在两片之间来回跳。这不影响 tracker 的输入。
  for (const auto &d : detected) {
    std::vector<cv::Point2f> one;
    one.reserve(d.keypoints.size());
    for (const auto &q : d.keypoints) one.emplace_back(q.x(), q.y());
    out.det_kpts.push_back(std::move(one));
    // 青环（旋转中心 R）：HW26 的 keypoints[8] 就是模型【直接输出】的转盘中心
    // 换回 HW26 后恢复这里，青环与"两片夹角"等几何判断才有依据。
    if (!detected.empty() && detected[0].keypoints.size() > 8) {
      const auto &rc = detected[0].keypoints[8];
      I.last_rcenter = cv::Point2f(rc.x(), rc.y());
      I.have_rcenter = true;
    }
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
    std::vector<std::size_t> pass;
    for (std::size_t i = 0; i < dets.size(); ++i) {
      const auto &b = dets[i];
      if (I.cfg.filter_color && b.color != I.cfg.buff_color) continue;
      if (I.cfg.filter_activation &&
          b.label != static_cast<int>(buff_algo::BuffActivation::INACTIVATE))
        continue;
      pass.push_back(i);
    }
    // 大符最多高亮 2 片，多于 2 个时按置信度取前 2（第 3 个通常是假阳）
    if (pass.size() > 2) {
      std::partial_sort(pass.begin(), pass.begin() + 2, pass.end(),
                        [&](std::size_t x, std::size_t y) {
                          return dets[x].confidence > dets[y].confidence;
                        });
      pass.resize(2);
    }
    for (std::size_t i : pass) {
      selected.push_back(dets[i]);
      selected_src.push_back(i);
    }
    if (!selected.empty()) out.dbg_conf0 = static_cast<double>(selected[0].confidence);
    if (selected.size() >= 2) out.dbg_conf1 = static_cast<double>(selected[1].confidence);
  } else if (I.cfg.filter_color && I.cfg.filter_activation) {
    I.selector->select_buffs(dets, selected);  // HW26 原逻辑（小符：只保留 1 个）
    if (!selected.empty()) out.dbg_conf0 = static_cast<double>(selected[0].confidence);
    if (selected.size() >= 2) out.dbg_conf1 = static_cast<double>(selected[1].confidence);
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

  //    刻意不调用 I.solver->solve_pnp()：HW26 的 BuffPnPSolver 内部写死
  //    consts::BUFF_POINTS = 0.114，那是它实车靶板的尺寸，用在模拟器上会让
  //    距离偏大约 10%（实测 10.18% vs 0.127 的 0.18%）。
  {
    int picks[2] = {-1, -1};
    int pi = 0;
    std::vector<std::vector<Pose3f>> cand_sets;   // 与 poses 平行（只含 PnP 成功的）
    for (std::size_t si = 0; si < selected.size(); ++si) {
    const auto &b = selected[si];
      int pk = -1;
      std::vector<Pose3f> cands;
      if (const auto pose =
              solve_pose_ippe(cv::Point2f(b.t.x(), b.t.y()), cv::Point2f(b.l.x(), b.l.y()),
                              cv::Point2f(b.b.x(), b.b.y()), cv::Point2f(b.r.x(), b.r.y()), &pk,
                              &cands)) {
        poses.push_back(*pose);
        cand_sets.push_back(std::move(cands));
        // ★ 与 poses 同步记录：只有 PnP 成功的才 push，两边永远一一对应
        pose_src.push_back(selected_src.size() > si ? selected_src[si] : si);
        if (pi < 2) picks[pi++] = pk;
      }
    }
    for (int i = 0; i < 2; ++i) out.sol_pick_t[i] = picks[i];

    // ★★ 两片联合消歧（IPPE 双解的正确打开方式）★★
    //   两个靶标刚性固连在同一转盘上，因此正确的那对 IPPE 解必须同时满足：
    //     ① 面内弦长落在相邻槽位 {0.823, 1.331} m 之一；
    //     ② 两片共面（沿符面法向的坐标相同）。
    //   法向 n 用【上一帧平滑后的符面法向】（符静止 -> 它是常量，与本次选择无关，
    //   所以不存在"用被选解反过来定义平面"的循环依赖）。
    //   这是逐帧、无历史依赖的离散选择：4 个组合里只有一个是物理自洽的，
    //   从根上掐掉 IPPE 翻转造成的相位跳变（实测 rcd 可达 0.89 m、dres 达 35°）。
    if (I.cfg.dual_target && I.face_ready && cand_sets.size() >= 2 && poses.size() >= 2 &&
        !cand_sets[0].empty() && !cand_sets[1].empty()) {
      const Eigen::Vector3f n = I.n_face;
      const double R = static_cast<double>(buff_algo::consts::BUFF_RADIUS);
      const double slots[2] = {2.0 * R * std::sin(0.5 * 2.0 * M_PI / 5.0),
                               2.0 * R * std::sin(1.0 * 2.0 * M_PI / 5.0)};
      double best = 1e18;
      int b0 = -1, b1 = -1;
      for (std::size_t i = 0; i < cand_sets[0].size(); ++i) {
        for (std::size_t j = 0; j < cand_sets[1].size(); ++j) {
          const Eigen::Vector3f t0 = R_cg * cand_sets[0][i].translation;
          const Eigen::Vector3f t1 = R_cg * cand_sets[1][j].translation;
          const double o0 = t0.dot(n), o1 = t1.dot(n);
          const Eigen::Vector3f a = t0 - static_cast<float>(o0) * n;
          const Eigen::Vector3f bq = t1 - static_cast<float>(o1) * n;
          const double d = (a - bq).norm();
          const double ds = std::min(std::abs(d - slots[0]), std::abs(d - slots[1]));
          const double cost = ds + 0.5 * std::abs(o0 - o1);
          if (cost < best) { best = cost; b0 = static_cast<int>(i); b1 = static_cast<int>(j); }
        }
      }
      if (b0 >= 0 && b1 >= 0) {
        const Eigen::Vector3f t0 = R_cg * cand_sets[0][b0].translation;
        const Eigen::Vector3f t1 = R_cg * cand_sets[1][b1].translation;
        const double o0 = t0.dot(n), o1 = t1.dot(n);
        const double d = ((t0 - static_cast<float>(o0) * n) -
                          (t1 - static_cast<float>(o1) * n)).norm();
        const double ds = std::min(std::abs(d - slots[0]), std::abs(d - slots[1]));
        if (ds < 0.12 && std::abs(o0 - o1) < 0.20) {   // 自洽才采纳
          poses[0] = cand_sets[0][b0];
          poses[1] = cand_sets[1][b1];
          out.sol_pick_t[0] = b0;
          out.sol_pick_t[1] = b1;
        }
      }
    }
  }
  }  // end if ext_kpts ... else（合成自检 vs 真实推理）
  }

  // 6-pre) 把本帧解出的位姿存成"上一帧位姿"，供下一帧的双解连续性使用。
  //         存全部（最多 2 片），不做身份绑定。
  I.prev_ts.clear();
  I.prev_qs.clear();
  for (const auto &q : poses) {
    if (I.prev_ts.size() >= 2) break;
    I.prev_ts.push_back(q.translation);
    I.prev_qs.push_back(q.rotation);
  }

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
    // 观测系 -> 车体系（meas_* 是观测系里的板心，画图前必须转对）
    auto obs2ch_v = [&](const Eigen::Vector3f &q) -> Eigen::Vector3f {
      if (I.cfg.frame == RefFrame::kFace) return I.R_chassis_face * q;
      if (I.cfg.frame == RefFrame::kChassis) return q;
      return R_cg * q;
    };
    // 6f-0) 转盘圆心 C
    //
    // C 是"相位 -> 打击点"的唯一基准：roll = atan2(t.z-C.z, C.y-t.y)，
    // 预测点 = (C.x, C.y - R cosθ, C.z + R sinθ)。C 错了，一切都跟着错。
    //
    // 为什么不能直接用 PnP 朝向反推：HW26 的 R_center = R*(0,0,-0.7) + t 依赖靶板的
    // 【面内朝向】，而 PnP 的朝向由 4 个点 t/l/b/r 恢复 —— 这 4 个点构成【正方形】，
    // 绕法线转 90° 自己重合，所以面内朝向有 4 重退化。实测两片由此解出的圆心分歧
    // 稳定在 0.8~1.6 m（dlt=144 时约 1.4 = 2*0.7；dlt=72 时约 0.85 = 2*0.7*sin36）。
    // 位置 t 没有这个退化，所以 C 只能靠"位置 + 半径 0.7 硬约束"来定。
    //
    // 两种来源（cfg.center_source）：
    //   0 = dual：|t_A-C| = |t_B-C| = 0.7 -> C 在 t_A t_B 的中垂线上，距中点
    //             h = sqrt(0.7² - (d/2)²)，两支用"模型输出的中心"选。
    //             好处是第一帧就有；坏处是**另一片的位置会进到这一片的结算里**
    //             （实测两片的面外间距 rcd 可达 0.60 m）。
    //   1 = single_history：只用被选中那一片的【历史 t】做固定半径圆拟合（见文件
    //             顶部的 fit_fixed_radius_center）。切断互相污染，还能剔掉该片
    //             自己 IPPE 翻转的野值；样本不够时自动回退到 0。
    Pose3f use_a = use;
    Pose3f use_b;
    bool have_b = false;
    if (I.cfg.dual_target && poses.size() >= 2) {
      use_b = to_obs(poses[1]);
      have_b = true;
    }
    const float r = buff_algo::consts::BUFF_RADIUS;
    const bool is_face = (I.cfg.frame == RefFrame::kFace);
    Eigen::Vector3f t0 = Eigen::Vector3f::Zero(), t1 = Eigen::Vector3f::Zero();
    const bool have_t1 = is_face && poses.size() >= 2;
    if (is_face && !poses.empty()) {
      t0 = Rc2f * (R_cg * poses[0].translation);
      if (have_t1) {
        t1 = Rc2f * (R_cg * poses[1].translation);
        // 弦长：不依赖任何朝向的独立观测量。相邻槽位(72°) 应 0.823 m，
        // 隔一槽位(144°) 应 1.331 m。两片被 IPPE 翻转拉开时它会明显偏离。
        out.chord_m = static_cast<double>((t0 - t1).norm());
      }
    }
    // 合成观测：z 轴取 (t - C)/r —— 天然落在符面内（x=0），于是 HW26 的固定平面
    // 前提【自动满足】，且 Buffalo 算出的 R_center 就等于 C（与朝向无关）。
    auto mk_obs = [&](const Eigen::Vector3f &tt) -> Pose3f {
      Eigen::Vector3f z = (tt - I.center_face) / r;
      z.x() = 0.f;
      if (z.norm() < 1e-3f) z = Eigen::Vector3f(0.f, 1.f, 0.f);
      z.normalize();
      const Eigen::Vector3f lx = Eigen::Vector3f::UnitX();
      Eigen::Vector3f ly = z.cross(lx);
      ly.normalize();
      Eigen::Matrix3f Rf;
      Rf.col(0) = lx;
      Rf.col(1) = ly;
      Rf.col(2) = z;
      Pose3f o;
      o.translation = tt;
      o.rotation = Eigen::Quaternionf(Rf).normalized();
      return o;
    };

    bool center_ok = false;
    if (is_face && !poses.empty()) {
      // ---- 来源 1：只用一片的历史做固定半径圆拟合 ----
      if (I.cfg.center_source == 1) {
        I.center_hist.emplace_back(t_rel, R_cg * poses[0].translation);
        while (I.center_hist.size() > 400) I.center_hist.pop_front();
        Eigen::Vector3f c;
        if (fit_fixed_radius_center(I.center_hist, Rc2f, r, I.cfg.center_hist_min,
                                    I.cfg.center_hist_span_m, I.cfg.center_hist_gate_m, t_rel,
                                    I.cfg.center_hist_keep_s, &c, &I.center_fit_n,
                                    &I.center_fit_rms_m)) {
          I.center_face = c;
          I.have_center = true;
          center_ok = true;
          out.center_src = 1;
        }
      }
      // ---- 来源 0（也是来源 1 的兜底）：两片位置联立 ----
      if (!center_ok && have_t1) {
        const Eigen::Vector2f a(t0.y(), t0.z()), b(t1.y(), t1.z());
        const Eigen::Vector2f d2 = b - a;
        const float dlen = d2.norm();
        if (dlen > 1e-4f && dlen <= 2.f * r) {
          const Eigen::Vector2f perp(-d2.y() / dlen, d2.x() / dlen);
          const float hh = std::sqrt(std::max(0.f, r * r - 0.25f * dlen * dlen));
          const Eigen::Vector2f mid = 0.5f * (a + b);
          const Eigen::Vector2f c1 = mid + hh * perp;
          const Eigen::Vector2f c2 = mid - hh * perp;
          // 选支：**不再用"离上一帧圆心更近"** —— 那是双稳态，错一次就锁死在错的
          // 一支，两片相位整体偏 180 度（实测 ra 从 +1.571 跳到 -1.574，差正好 pi）。
          // 改成：把两个候选都投到图像，取离模型输出的转盘中心（keypoints[8]）更近的。
          // 两个候选在图像上相距约 57~293 像素，只需二选一，模型中心几厘米的偏差
          // 不影响判定；而精确圆心仍由几何解（两片 + 半径 0.7 硬约束）给出。
          Eigen::Vector2f cc = c1;
          bool picked_by_model = false;
          if (I.have_rcenter) {
            auto proj_cand = [&](const Eigen::Vector2f &cyz, int *px, int *py) -> bool {
              const Eigen::Vector3f cf(0.5f * (t0.x() + t1.x()), cyz.x(), cyz.y());
              const Eigen::Vector3f pc = R_cg.transpose() * (I.R_chassis_face * cf);
              return project_cam(gimbal_to_cam(pc), I.cfg.camera_matrix, px, py);
            };
            int x1 = -1, y1 = -1, x2 = -1, y2 = -1;
            if (proj_cand(c1, &x1, &y1) && proj_cand(c2, &x2, &y2)) {
              const double d1 = std::hypot(static_cast<double>(x1) - I.last_rcenter.x,
                                           static_cast<double>(y1) - I.last_rcenter.y);
              const double d2m = std::hypot(static_cast<double>(x2) - I.last_rcenter.x,
                                            static_cast<double>(y2) - I.last_rcenter.y);
              cc = (d2m < d1) ? c2 : c1;
              out.cand_px[0] = d1;
              out.cand_px[1] = d2m;  // 诊断：判别余量应该很大（否则这个方法不可靠）
              picked_by_model = true;
            }
          }
          if (!picked_by_model && I.have_center) {
            // 兜底：拿不到模型中心时才退回"离上一帧更近"（已知是双稳态，尽量别用）
            const Eigen::Vector2f pc(I.center_face.y(), I.center_face.z());
            if ((c2 - pc).norm() < (c1 - pc).norm()) cc = c2;
          } else if (!picked_by_model) {
            // 首帧：用朝向推出的圆心（取两片平均）当中垂线两支解的初值
            const Eigen::Vector3f c_ori =
                0.5f * (Eigen::Vector3f(Buff(use).R_center) +
                        Eigen::Vector3f(Buff(use_b).R_center));
            const Eigen::Vector2f pc(c_ori.y(), c_ori.z());
            if ((c2 - pc).norm() < (c1 - pc).norm()) cc = c2;
          }
          I.center_face = Eigen::Vector3f(0.5f * (t0.x() + t1.x()), cc.x(), cc.y());
          I.have_center = true;
          center_ok = true;
        }
      }
      // ★ 只要拿到过圆心（have_center）就用 mk_obs 合成观测，而不是等本帧 center_ok。
      //   否则 obs.roll / dbg_roll_* 会在【圆心系 roll】和【原始 PnP roll】两套约定
      //   之间随 center_ok 切换，这本身就是相位源跳变（rms 差的窗里 om 顶边界、span 不小）。
      if (I.have_center) {
        use_a = mk_obs(t0);
        if (have_b) use_b = mk_obs(t1);
      }
    }
    out.center_n = (I.cfg.center_source == 1) ? I.center_fit_n : 0;
    out.center_rms_m = (I.cfg.center_source == 1) ? I.center_fit_rms_m : -1.0;
    if (I.cfg.center_source != 1) out.center_src = 0;

    // 6f) 双靶标融合（大符）
    //   两片高亮靶标刚性固连在同一个转盘上，所以 roll_B(t) = roll_A(t) + Δ，
    //   Δ 是常数（5 个槽位 -> 72° 的整数倍）。于是同一帧的两个检出是对【同一个】
    //   三角函数体系的两次独立测量，可以融合成一次、噪声降 sqrt(2)；
    //   而且融合后的序列连续，不会再触发 switch_buff_angle 的跳变判据。
    Buff obs(use_a);
    out.rc_dist_m = static_cast<double>(obs.R_center.norm());
    out.dual_targets = 1;
    out.meas_a_valid = true;
    out.meas_a_ch = obs2ch_v(use_a.translation);  // 1 号实测板心（车体系）
    if (I.cfg.dual_target && have_b) {
      const Pose3f use2 = use_b;
      // ★ 组号必须【稳定】：按 wrap 之后的相位先后决定谁是 1 号，而不是按置信度。
      //   按置信度排的话，两片置信度互相超过时（0.91 <-> 0.89）黄/品红会来回换 ——
      //   这是实测看到的现象。相位差是常数，所以 wrap 后的先后是稳定的。
      Buff A = obs;
      Buff B(use2);
      bool pose0_is_a = true;
      if (wrap_pi(static_cast<double>(B.roll) - static_cast<double>(A.roll)) < 0.0) {
        std::swap(A, B);
        pose0_is_a = false;
      }
      const std::size_t src_a = pose_src.size() >= 2
                                    ? (pose0_is_a ? pose_src[0] : pose_src[1])
                                    : 0;
      const std::size_t src_b = pose_src.size() >= 2
                                    ? (pose0_is_a ? pose_src[1] : pose_src[0])
                                    : 1;
      out.dbg_pose0_is_a = pose0_is_a ? 1 : 0;
      out.dbg_src_a = static_cast<int>(src_a);
      out.dbg_src_b = static_cast<int>(src_b);
      const double step = 2.0 * M_PI / 5.0;
      const double d = wrap_pi(static_cast<double>(B.roll) - static_cast<double>(A.roll));  // >= 0
      int k = static_cast<int>(std::lround(d / step));
      if (k == 0) k = 1;  // 两靶标不可能同相位
      const double delta = k * step;
      // 诊断：吸附残差与两中心分歧 —— 这两条是"72° 刚性固连"假设的直接检验
      out.dual_resid_deg = std::abs(d - delta) * 180.0 / M_PI;
      out.dual_rcd_m = static_cast<double>((A.R_center - B.R_center).norm());
      // 【不再融合两片】直接喂 A 自己的相位。
      // 理由：正弦模型本来就是对【整个窗口】做最小二乘，自己就会平均噪声，
      // 不需要靠两片融合降噪；而融合依赖 Δ 恰好是 72/144 的整数倍
      // （实测 dres 还有 3~23 度），反而把这个误差混进被喂的相位里。
      // 现在拟合的是【一片真实叶片】的相位，A.R_center 也保持原样（不做平均）。
      obs = A;
      out.dual_targets = 2;
      out.dual_delta_deg = delta * 180.0 / M_PI;
      // 诊断：融合【前】的原始相位 + 各自的 R_center 距离
      out.dbg_roll_a = static_cast<double>(A.roll);
      out.dbg_roll_b = static_cast<double>(B.roll);
      out.dbg_rca_m = static_cast<double>(A.R_center.norm());
      out.dbg_rcb_m = static_cast<double>(B.R_center.norm());
      cur_rolls = {A.roll, B.roll};
      out.meas_a_valid = true;
      out.meas_b_valid = true;
      out.meas_a_ch = obs2ch_v(pose0_is_a ? use.translation : use2.translation);
      out.meas_b_ch = obs2ch_v(pose0_is_a ? use2.translation : use.translation);
      // 关键点组号【跟着 A/B 走】：组 0（黄）= A = 亮绿点，组 1（品红）= B = 暗绿点。
      //
      // 原来这里是 std::swap(out.det_kpts[src_a], out.det_kpts[src_b])，那是错的：
      // 它只交换两个来源槽的内容，等价于"把 A 放到 0 号槽"【当且仅当 src_a == 1】。
      // 而 selected_src 是跟着【置信度】排的（两片 0.91 <-> 0.89 一直在互相超过），
      // src_a 会在 0/1 之间跳 -> 黄/品红于是逐帧反转，而绿点（直接按 A/B 取
      // translation）一直是对的。所以"颜色在换"是绘制 bug，不是跟踪在跳。
      // 改成按 A、B 顺序重建列表，第 3 组起（假阳）跟在后面。
      if (out.det_kpts.size() >= 2 && src_a != src_b &&
          src_a < out.det_kpts.size() && src_b < out.det_kpts.size()) {
        std::vector<std::vector<cv::Point2f>> ordered;
        ordered.reserve(out.det_kpts.size());
        ordered.push_back(out.det_kpts[src_a]);  // 组 0 = A
        ordered.push_back(out.det_kpts[src_b]);  // 组 1 = B
        for (std::size_t gi = 0; gi < out.det_kpts.size(); ++gi) {
          if (gi != src_a && gi != src_b) ordered.push_back(out.det_kpts[gi]);
        }
        out.det_kpts = std::move(ordered);
      }
    }
    // 6f-0b) 喂桥自己的局部相位 tracker（用【滤波前】的原始相位：
    //        它内部本来就是滑动窗口最小二乘，会自己平滑；喂平滑过的会叠加滞后）。
    if (I.cfg.local_tracker) {
      // 让 绿点 = 橙点的预测 成为恒等式：
      //   以前把 PnP 的 roll 直接喂给局部 tracker，而橙点 meas_a_ch 是 PnP 解出的板心位置，
      //   绿点又是理想圆上按 theta 外推的点 —— 两条不同几何路径，两者会偏几十像素。
      //   现在改成：用橙点自己的位置反解出它绕转盘中心的角度 theta_A 再喂进去。
      //   绿点公式 y = C.y - R*cos(theta), z = C.z + R*sin(theta)
      //   => theta = atan2(z - C.z, C.y - y)
      //   所以 predict() 的 theta_A(t+lead) 代回同一公式，就恒等于橙点 lead 秒后的位置。
      double theta_feed = static_cast<double>(obs.roll);   // 兜底：中心或板心无效时用原 roll
      bool theta_from_chord = false;
      bool theta_fused = false;
      bool theta_roll = false;
      // ★★★ 两片 roll 融合 + 【统一约定】（默认开）★★★
      //   两片刚性固连同一转盘 -> roll_B = roll_A + Δ (Δ = k·72°)。
      //   对齐后平均：噪声降 sqrt(2)。
      //   关键：一旦走 roll 约定，就【整段都用 roll 约定】，绝不退回圆心路径
      //   （实测混源时源切换 617 次，切换帧 |Δθ| p90=147°，而不切换帧只有 7.6°）。
      //   roll 约定下 A/B 身份互换是常数平移，用"离上一帧最近"消歧即可。
      if (I.cfg.phase_fuse_dual) {
        theta_roll = true;
        if (out.dual_targets == 2) {
          const double delta = out.dual_delta_deg * M_PI / 180.0;
          const double ca = out.dbg_roll_a;
          const double cb = wrap_pi(out.dbg_roll_b - delta);   // B 折到 A 的槽位
          const double resid = wrap_pi(cb - ca);
          if (std::abs(resid) * 180.0 / M_PI < 20.0) {
            theta_feed = wrap_pi(ca + 0.5 * resid);            // 一致 -> 平均
            theta_fused = true;
          } else if (I.have_last_local_theta && I.local_trk.n() > 0) {
            theta_feed = (std::abs(wrap_pi(cb - I.last_local_theta)) <
                          std::abs(wrap_pi(ca - I.last_local_theta))) ? cb : ca;
          } else {
            theta_feed = ca;
          }
        }
        // 单片：theta_feed 已经是 obs.roll，属于同一 Buff roll 约定，直接用。
      }
      // ★★★ 弦相位（创新，圆心无关）★★★
      //   两片点亮的靶标刚性固连在同一转盘上，所以它们的连线向量也随转盘【刚体旋转】：
      //       chord = p_B - p_A ∝ (sin η, cos η),  η = θ_A + Δ/2,  Δ = k·72°
      //   => χ = atan2(chord_z, chord_y) = π/2 - η  =>  η = π/2 - χ
      //   => 两个候选 θ = η ∓ Δ/2（它们都是真实点亮的靶标，任取其一都能打）。
      //   好处：① 完全不需要圆心 C（绕开 §1.10 那套固定半径圆拟合及其误差）；
      //        ② 两片位置做差，公共误差（距离/尺度偏差）自动抵消；
      //        ③ 身份换片只会让候选整体平移 Δ，用"离上一帧最近"即可稳定消歧，
      //           不会再像 PnP roll 那样每帧跳 180°/72°。
      //   Δ 由弦长自身给出（对朝向免疫）：chord = 2R sin(Δ/2)。
      if (!theta_roll && !theta_fused && I.cfg.chord_phase && is_face && have_t1 &&
          out.meas_a_valid) {
        const double dy = static_cast<double>(t1.y() - t0.y());
        const double dz = static_cast<double>(t1.z() - t0.z());
        const double chord_len = std::hypot(dy, dz);
        const double step = 2.0 * M_PI / 5.0;
        const double s = std::min(1.0, chord_len / (2.0 * static_cast<double>(r)));
        int kk = static_cast<int>(std::lround(2.0 * std::asin(s) / step));
        if (kk < 1) kk = 1;
        if (kk > 2) kk = 2;
        const double delta = kk * step;
        const double chord_exp = 2.0 * static_cast<double>(r) * std::sin(0.5 * delta);
        if (chord_len > 0.15 && std::abs(chord_len - chord_exp) < 0.18) {
          const double eta = 0.5 * M_PI - std::atan2(dz, dy);
          const double c1 = wrap_pi(eta - 0.5 * delta);
          const double c2 = wrap_pi(eta + 0.5 * delta);
          const double ref = (I.have_last_local_theta && I.local_trk.n() > 0)
                                 ? I.last_local_theta
                                 : static_cast<double>(obs.roll);
          theta_feed = (std::abs(wrap_pi(c2 - ref)) < std::abs(wrap_pi(c1 - ref))) ? c2 : c1;
          theta_from_chord = true;
        }
      }
      if (!theta_roll && !theta_fused && !theta_from_chord && out.meas_a_valid) {
        // 单片兜底：还是用圆心（老路径），保证单片也能工作
        const Eigen::Vector3f p_face = I.R_chassis_face.transpose() * out.meas_a_ch;
        const double dy = static_cast<double>(I.center_face.y() - p_face.y());
        const double dz = static_cast<double>(p_face.z() - I.center_face.z());
        if (std::hypot(dy, dz) > 1e-3) theta_feed = std::atan2(dz, dy);
      }
      I.last_local_theta = theta_feed;
      I.have_last_local_theta = true;
      out.local_theta = theta_feed;
      out.local_chord = theta_roll ? 2 : (theta_from_chord ? 1 : 0);
      out.local_phys_rej = I.local_trk.phys_rej();
      I.local_trk.push(t_rel, theta_feed);
      out.local_n = I.local_trk.n();
      out.local_ready = I.local_trk.ready() ? 1 : 0;
      out.local_w = I.local_trk.velocity();
      out.local_rms_deg = I.local_trk.rms_deg();
      out.local_locked = I.local_trk.locked() ? 1 : 0;
      out.local_omega = I.local_trk.omega();
      out.local_amp = I.local_trk.amp();
      out.local_span_deg = I.local_trk.span_deg();
      out.local_drop = I.local_trk.jump_dropped();
      out.local_lost = I.local_trk.lost() ? 1 : 0;
      out.local_hit = I.local_trk.hit_confirmed() ? 1 : 0;
    }

    // 6f-1) 相位低通（滑动平均 N 帧，解缠后平均）。目的见 Config::phase_ma_n 注释。
    //       只在连续有观测的帧之间累计；一旦断帧（det=0）就清空，避免跨窗口平均。
    if (I.cfg.phase_ma_n > 1) {
      double raw = static_cast<double>(obs.roll);
      double unw = raw;
      if (I.have_last_roll) {
        unw = I.last_roll_unwrapped + wrap_pi(raw - I.last_roll_raw);
      }
      I.last_roll_raw = raw;
      I.last_roll_unwrapped = unw;
      I.have_last_roll = true;
      I.roll_hist.push_back(unw);
      while (static_cast<int>(I.roll_hist.size()) > I.cfg.phase_ma_n) I.roll_hist.pop_front();
      double acc = 0.0;
      for (double v : I.roll_hist) acc += v;
      obs.roll = static_cast<float>(wrap_pi(acc / static_cast<double>(I.roll_hist.size())));
    }
    // 诊断：重排后组 0 / 组 1 的关键点质心（图像像素），以及两组置信度
    if (!out.det_kpts.empty() && !out.det_kpts[0].empty()) {
      double sx = 0.0, sy = 0.0;
      for (const auto &q : out.det_kpts[0]) { sx += q.x; sy += q.y; }
      out.dbg_g0x = sx / static_cast<double>(out.det_kpts[0].size());
      out.dbg_g0y = sy / static_cast<double>(out.det_kpts[0].size());
    }
    if (out.det_kpts.size() >= 2 && !out.det_kpts[1].empty()) {
      double sx = 0.0, sy = 0.0;
      for (const auto &q : out.det_kpts[1]) { sx += q.x; sy += q.y; }
      out.dbg_g1x = sx / static_cast<double>(out.det_kpts[1].size());
      out.dbg_g1y = sy / static_cast<double>(out.det_kpts[1].size());
    }
    if (dis_ok) I.tracker->push(obs);

      // 6g) 诊断：两片 roll 的帧间变化（不依赖身份的最优配对）。
      //     某一对残差接近 180 度 = 那片发生了 IPPE 双解翻转 -> 相位被翻到对面。
      if (cur_rolls.size() == 2 && I.prev_rolls.size() == 2) {
        auto wa = [](double a) { return wrap_pi(a); };
        const double a00 = std::abs(wa(cur_rolls[0] - I.prev_rolls[0]));
        const double a11 = std::abs(wa(cur_rolls[1] - I.prev_rolls[1]));
        const double a01 = std::abs(wa(cur_rolls[0] - I.prev_rolls[1]));
        const double a10 = std::abs(wa(cur_rolls[1] - I.prev_rolls[0]));
        const bool straight = (std::max(a00, a11) <= std::max(a01, a10));
        out.droll_deg[0] = (straight ? a00 : a01) * 180.0 / M_PI;
        out.droll_deg[1] = (straight ? a11 : a10) * 180.0 / M_PI;
      }
      I.prev_rolls = cur_rolls;
  }

  // 帧间隔（EMA）——用于把相位滤波的滞后换算成时间
  if (I.have_prev_frame_t) {
    const double dt = t_rel - I.prev_frame_t;
    if (dt > 1e-4 && dt < 0.5) I.dt_frame_ema = 0.8 * I.dt_frame_ema + 0.2 * dt;
  }
  I.prev_frame_t = t_rel;
  I.have_prev_frame_t = true;
  // 断了观测就把相位历史清空（下次点亮重新开始，不跨窗口平均）
  if (out.dual_targets == 0) {
    I.roll_hist.clear();
    I.have_last_roll = false;
    I.local_trk.on_dark(t_rel);   // 记录灭灯时刻（用于"连续无空窗=命中"判定）
  }

  // 7) 状态机推进（无论本帧是否有观测都要 update，与 HW26 example 一致）
  I.tracker->update(t_rel);
  out.status = static_cast<int>(I.tracker->status());
  const BuffState state = I.tracker->get_state();
  out.roll = state.roll;
  out.roll_velocity = state.roll_velocity;

  // 8) 预测 + 弹道 + 云台角
  const bool trackable = (out.status == static_cast<int>(StatusType::TRACKING));
  double lead = 0.0, fly = 0.0, pitch_need = 0.0;
  Eigen::Vector3f aim_pt = Eigen::Vector3f::Zero();
  bool aim_valid = false;

  const bool local_usable =
      (I.cfg.local_tracker != 0) && I.local_trk.ready() && I.have_center;
  // 【灭灯也要继续跟踪】灭灯(det=0)时没有 PnP -> out.solved 为假，但 mode6 的滑行仍在
  // 有效外推（参数冻结、t_ref 不变 => 相位照常前进）。不放行的话绿点会在灭灯期消失，
  // 看起来就是"一个一个的绿点"而不是连续轨迹。
  // 【局部 tracker 的预测不需要本帧的 PnP】
  //   它只需要：窗口就绪 + 转盘中心（两者都在亮灯期锁存好了）。
  //   原来把 out.solved 当硬条件 -> PnP 失败的那些帧（实测约占一半：solved=0 3261 帧 /
  //   solved=1 3158 帧）整段瞄准被跳过 -> 绿点就变成"一个一个"的闪烁 ✗。
  //   现在：局部可用即可进（滑行时 ready() 仍为真，所以也包含灭灯期）。
  const bool local_ok =
      (I.cfg.local_tracker != 0) && I.local_trk.ready() && I.have_center;
  // ★ 本地 ready 时优先用本地；aim_local_only=1 时本地未 ready 就【不出瞄准】，
  //   避免回退到会转圈的 HW26 predictor（实测 438/682 控制帧来自它）。
  const bool local_only = I.cfg.aim_local_only && (I.cfg.local_tracker != 0);
  if (local_ok || (!local_only && out.solved && trackable)) {
    const double alg_lag =
        (I.cfg.alg_latency > 0.0) ? I.cfg.alg_latency : I.last_latency_ms / 1000.0;
    // 相位滤波带来的滞后：滑动平均 N 帧 -> 相位代表的是 (N-1)/2 帧之前的时刻，
    // 所以预测时间要把这段补回去，否则会系统性晚 (N-1)/2*dt（N=6 时约 85ms）。
    const double ma_lag =
        (I.cfg.phase_ma_n > 1)
            ? 0.5 * static_cast<double>(I.cfg.phase_ma_n - 1) * I.dt_frame_ema
            : 0.0;
    const double base_lead =
        alg_lag + I.cfg.predict_time + I.cfg.gimbal_lead_time + ma_lag;

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
      // 预测命中点：HW26 到 TRACKING 就用它的 predictor（4 参数正弦外推）；
      // 否则用桥自己的局部 tracker（相位二次外推，6~8 个样本即可用）。
      // 优先级翻转：本地 mode7 ready 就用本地（它带 rms 门，可信）；
      // 只有本地不可用、且没开 local_only 时才回退 HW26 predictor。
      if (local_ok) {
        double roll_pred = 0.0;
        if (!I.local_trk.predict(t_rel + lead, &roll_pred)) {
          aim_valid = false;
          break;
        }
        const float R = buff_algo::consts::BUFF_RADIUS;
        aim_pt.x() = I.center_face.x();
        aim_pt.y() = I.center_face.y() - R * std::cos(static_cast<float>(roll_pred));
        aim_pt.z() = I.center_face.z() + R * std::sin(static_cast<float>(roll_pred));
      } else if (out.status == static_cast<int>(StatusType::TRACKING)) {
        // 仅当没开 local_only 且本地不可用时才回退（会转圈的旧路径）
        aim_pt = I.predictor->predict_position(static_cast<float>(lead));
      } else {
        aim_valid = false;
        break;
      }
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

      // (1) aim 像素瞬移拒帧：PnP 翻转 / 换扇叶会让 aim 点瞬移 -> 本帧不伺服
      bool aim_jumped = false;
      if (I.have_prev_aim_px && out.aim_px_x > 0) {
        const double step = std::hypot(static_cast<double>(out.aim_px_x) - I.prev_aim_px.x,
                                       static_cast<double>(out.aim_px_y) - I.prev_aim_px.y);
        aim_jumped = step > I.cfg.max_aim_step_px;
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
      // ★ 预测可信才允许开火：模型 ready、拟合 rms 小、样本够。
      //   命中后换组 -> 模型门连续拒绝 -> reset -> ready=0，于是自动停火等重锁，
      //   不再"打中了还继续往外打"。
      const bool pred_confident =
          !I.cfg.fire_require_lock ||
          ((out.local_ready != 0) && out.local_rms_deg >= 0.0 &&
           out.local_rms_deg <= I.cfg.fire_rms_gate_deg && out.local_n >= I.cfg.fire_min_n);
      if (I.cfg.fire_enabled && pred_confident &&
          I.align_run >= I.cfg.fire_stable_frames &&
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
          project_px(muzzle_ch + tt * dir_ch, &out.blue_px_x, &out.blue_px_y);
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

    // ---- 紫点：未来打击点（预测命中点）----
    // 颜色按用户要求由绿改紫（BGR 240,32,160 = RGB 160,32,240）；
    // 按用户要求改成【空心】且直径更大：r=14（直径 28，原 r=8 实心）、线宽 3。
    // 空心时判读依据是【圆环中心】。
    // 注意：本文件其余注释与日志里仍把它叫"绿点"，指的是同一个点。
    if (out.green_px_x > 0) {
      cv::circle(out.debug, {out.green_px_x, out.green_px_y}, 14, cv::Scalar(240, 32, 160), 3,
                 cv::LINE_AA);
    }
    // ---- 蓝点（枪管射线 ∩ 符平面）已按用户要求【取消绘制】----
    // 数据仍然计算（out.blue_px_* 保留，供日志与调试用），只是不画。

    // ---- 把检测到的东西画出来（让"在检测什么"可见）----
    // 多个检出各自一组颜色：第 1 组黄（与原行为一致），第 2 组品红，第 3 组起循环。
    // 以前只画 detected[0]，大符时会在两片靶标之间来回跳。
    {
      // 【一组一个颜色：组内六个点同色，两组之间颜色不同】
      //   组 1 = 黄，组 2 = 品红。半径统一（不再靠大小暗示身份），组号另用数字标出。
      static const cv::Scalar kGroupColor[2] = {
          cv::Scalar(0, 255, 255),   // 组 1：黄
          cv::Scalar(255, 0, 255),   // 组 2：品红
      };
      const std::size_t kDrawGroups = std::min<std::size_t>(2, out.det_kpts.size());
      for (std::size_t g = 0; g < kDrawGroups; ++g) {
        const cv::Scalar &col = kGroupColor[g];
        // ★ 按用户要求：每组只画【1 个点】= 第 9 个（index 8，模型直接输出的旋转中心）
      if (!out.det_kpts[g].empty()) {
        const cv::Point2f &d = out.det_kpts[g].back();
        cv::circle(out.debug, {static_cast<int>(d.x), static_cast<int>(d.y)}, 6, col,
                   cv::FILLED, cv::LINE_AA);
      }
        // 编号：默认 1 = A（相位靠前 = 先打的那片）；num_reverse=1 时两组编号对调。
        if (!out.det_kpts[g].empty()) {
          const cv::Point2f &p0 = out.det_kpts[g].back();
          const std::size_t shown = (I.cfg.num_reverse != 0) ? (kDrawGroups - g) : (g + 1);
          char tag[8];
          std::snprintf(tag, sizeof(tag), "%zu", shown);
          cv::putText(out.debug, tag, {static_cast<int>(p0.x) + 6, static_cast<int>(p0.y) - 6},
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv::LINE_AA);
        }
      }
    }
    if (I.have_rcenter) {
      cv::circle(out.debug, {static_cast<int>(I.last_rcenter.x), static_cast<int>(I.last_rcenter.y)},
                 10, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);  // 中心 R（青）
    }

    // ---- 两片各自实测的板心（亮绿/暗绿）----
    //   判据：两点应落在同一个转盘的轨道上，角间距 72° 或 144°；
    //   它们各自对应一片点亮的靶标（关键点组 1 / 组 2 也应该各落在一片上面）。
    if (I.face_ready) {
      // 两片各自实测的板心
      int ax = -1, ay = -1, bx = -1, by = -1;
      const bool okA = out.meas_a_valid && proj_ch(out.meas_a_ch, &ax, &ay);
      const bool okB = out.meas_b_valid && proj_ch(out.meas_b_ch, &bx, &by);
      // 两组各自的【三维板心重投影】，一色一组：
      //   组 1（六个黄点）的板心 -> 橙；组 2（六个品红点）的板心 -> 蓝。
      //   （蓝用 (255,80,0)：纯蓝 (255,0,0) 已被"枪管射线∩符平面"的蓝点占用，略作区分。）
      const cv::Scalar kCenterA(0, 165, 255);   // 橙 = 组1 的板心
      const cv::Scalar kCenterB(255, 80, 0);    // 蓝 = 组2 的板心
      if (okA) {
        cv::circle(out.debug, {ax, ay}, 6, kCenterA, cv::FILLED, cv::LINE_AA);
        // 组号写在板心点旁：应与第 1 组关键点上的"1"落在同一片靶标上
        cv::putText(out.debug, "1", {ax + 8, ay + 16}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    kCenterA, 2, cv::LINE_AA);
      }
      if (okB) {
        cv::circle(out.debug, {bx, by}, 6, kCenterB, cv::FILLED, cv::LINE_AA);
        // 暗绿点 = 组 2（B），应与品红关键点组上的"2"落在同一片靶标上
        cv::putText(out.debug, "2", {bx + 8, by + 16}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    kCenterB, 2, cv::LINE_AA);
      }
    }
    // ★ 按用户要求：HUD 的 5 行文字全部去掉（省下最贵的 putText）

  }

  g_stage_acc.mark(4);
  const auto t_end = std::chrono::steady_clock::now();
  out.lead_s = lead;
  out.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  I.last_latency_ms = out.latency_ms;
  g_stage_acc.report(out.latency_ms);

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
            << gimbal_roll * 180.0 / M_PI << ',' << out.face_tilt_deg << ','
            << out.local_theta << ',' << out.local_chord << '\n';
      I.log.flush();
    }
  }
  ++I.frame_count;
  (void)dt_frame;
  return out;
}

}  // namespace hw26_bridge
