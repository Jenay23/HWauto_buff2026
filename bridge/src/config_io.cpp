#include "config_io.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// ---- 极简扁平行配置解析：key: value / key: [a, b, c] / # 注释 ----
// 不用 cv::FileStorage，避免它对 YAML 方言的挑剔，配置也更好手改。
struct FlatConfig {
  std::map<std::string, std::string> kv;
  std::map<std::string, std::vector<double>> lists;

  bool has(const std::string &k) const { return kv.count(k) || lists.count(k); }
  int geti(const std::string &k, int d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    try { return std::stoi(it->second); } catch (...) { return d; }
  }
  double getd(const std::string &k, double d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    try { return std::stod(it->second); } catch (...) { return d; }
  }
  bool getb(const std::string &k, bool d) const {
    auto it = kv.find(k);
    if (it == kv.end()) return d;
    const std::string v = it->second;
    return !(v == "0" || v == "false" || v == "no" || v == "off");
  }
  std::string gets(const std::string &k, const std::string &d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
};

std::string trim(const std::string &s) {
  const size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

FlatConfig parse_flat(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  FlatConfig fc;
  std::string line;
  while (std::getline(in, line)) {
    const size_t h = line.find('#');
    if (h != std::string::npos) line = line.substr(0, h);
    line = trim(line);
    if (line.empty()) continue;
    const size_t c = line.find(':');
    if (c == std::string::npos) continue;
    const std::string key = trim(line.substr(0, c));
    std::string val = trim(line.substr(c + 1));
    if (key.empty()) continue;
    if (val.size() >= 2 && val.front() == '[' && val.back() == ']') {
      std::vector<double> nums;
      std::string cur;
      for (size_t i = 1; i + 1 < val.size(); ++i) {
        const char ch = val[i];
        if (ch == ',') {
          if (!trim(cur).empty()) nums.push_back(std::stod(trim(cur)));
          cur.clear();
        } else {
          cur.push_back(ch);
        }
      }
      if (!trim(cur).empty()) nums.push_back(std::stod(trim(cur)));
      fc.lists[key] = nums;
    } else {
      if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
        val = val.substr(1, val.size() - 2);
      }
      fc.kv[key] = val;
    }
  }
  return fc;
}


}  // namespace

BridgeConfig load_config(const std::string &path) {
  BridgeConfig bc;
  const FlatConfig fs = parse_flat(path);

  // 检测后端不再从配置读：只写一个 model: 路径，Pipeline 按模型的输出形状判定
  bc.algo.model_path = fs.gets("model", bc.algo.model_path);
  bc.algo.device = fs.gets("device", bc.algo.device);
  bc.algo.blade_half_size = fs.getd("blade_half_size", bc.algo.blade_half_size);
  bc.algo.blade_half_size_hw26 =
      fs.getd("blade_half_size_hw26", bc.algo.blade_half_size_hw26);
  bc.algo.use_muzzle_offset = fs.getb("use_muzzle_offset", bc.algo.use_muzzle_offset);
  const auto mo = fs.lists.count("muzzle_offset") ? fs.lists.at("muzzle_offset") : std::vector<double>{};
  if (mo.size() == 3) {
    bc.algo.muzzle_offset[0] = mo[0];
    bc.algo.muzzle_offset[1] = mo[1];
    bc.algo.muzzle_offset[2] = mo[2];
  }
  if (bc.algo.model_path.empty()) throw std::runtime_error("config: model path is required");
  // 相对路径按“配置文件所在目录”解析，避免依赖 CWD
  if (bc.algo.model_path.front() != '/') {
    const size_t slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
    bc.algo.model_path = dir + "/" + bc.algo.model_path;
  }

  const auto km = fs.lists.count("camera_matrix") ? fs.lists.at("camera_matrix")
                                                  : std::vector<double>{};
  if (km.size() == 9) {
    bc.algo.camera_matrix = (cv::Mat_<double>(3, 3) << km[0], km[1], km[2], km[3], km[4], km[5],
                             km[6], km[7], km[8]);
  }
  const auto dc = fs.lists.count("distort_coeffs") ? fs.lists.at("distort_coeffs")
                                                   : std::vector<double>{};
  if (dc.size() == 5) {
    bc.algo.distort_coeffs = (cv::Mat_<double>(1, 5) << dc[0], dc[1], dc[2], dc[3], dc[4]);
  }

  bc.algo.buff_mode = fs.geti("mode", 1);
  bc.algo.buff_color = fs.geti("my_color", 0);
  bc.algo.confidence_threshold = static_cast<float>(fs.getd("confidence_threshold", 0.5));
bc.algo.num_reverse = static_cast<int>(fs.getd("num_reverse", bc.algo.num_reverse));
  bc.algo.nms_threshold = static_cast<float>(fs.getd("nms_threshold", 0.4));
  bc.algo.filter_color = fs.getb("filter_color", true);
  bc.algo.filter_activation = fs.getb("filter_activation", true);
  bc.algo.dual_target = fs.getb("dual_target", true);
  bc.algo.chord_phase = fs.getb("chord_phase", false);
  bc.algo.phase_fuse_dual = fs.getb("phase_fuse_dual", true);
  bc.algo.phase_ma_n = static_cast<int>(fs.getd("phase_ma_n", bc.algo.phase_ma_n));
  bc.algo.local_tracker = static_cast<int>(fs.getd("local_tracker", bc.algo.local_tracker));
  bc.algo.local_model =
      static_cast<int>(fs.getd("local_model", bc.algo.local_model));
  bc.algo.local_rms_gate_deg = fs.getd("local_rms_gate_deg", bc.algo.local_rms_gate_deg);
  // A/橙 身份（由旋向决定）判准后锁定多久（秒）；默认 420 = 7 分钟
  bc.algo.dir_lock_s = fs.getd("dir_lock_s", bc.algo.dir_lock_s);
  bc.algo.local_bad_rms_deg = fs.getd("local_bad_rms_deg", bc.algo.local_bad_rms_deg);
  bc.algo.local_coast_s = fs.getd("local_coast_s", bc.algo.local_coast_s);
  bc.algo.center_source = static_cast<int>(fs.getd("center_source", bc.algo.center_source));
  bc.algo.center_hist_min = static_cast<int>(fs.getd("center_hist_min", bc.algo.center_hist_min));
  bc.algo.center_hist_span_m = fs.getd("center_hist_span_m", bc.algo.center_hist_span_m);
  bc.algo.center_hist_gate_m = fs.getd("center_hist_gate_m", bc.algo.center_hist_gate_m);
  bc.algo.pos_gate_m = fs.getd("pos_gate_m", bc.algo.pos_gate_m);
  bc.algo.center_hist_keep_s = fs.getd("center_hist_keep_s", bc.algo.center_hist_keep_s);
  bc.algo.bullet_speed = fs.getd("bullet_speed", 22.0);
  bc.algo.gravity = fs.getd("gravity", 9.81);
  bc.algo.yaw_offset_deg = fs.getd("yaw_offset_deg", 0.0);
  bc.algo.pitch_offset_deg = fs.getd("pitch_offset_deg", -20.0);
  bc.algo.predict_time = fs.getd("predict_time", 0.10);
  bc.algo.gimbal_lead_time = fs.getd("gimbal_lead_time", 0.0);
  bc.algo.alg_latency = fs.getd("alg_latency", 0.0);
  bc.algo.min_distance = fs.getd("min_distance", 2.0);
  bc.algo.max_distance = fs.getd("max_distance", 15.0);
  bc.algo.fire_err_rad = fs.getd("fire_err_rad", 0.02);
  bc.algo.fire_stable_frames = fs.geti("fire_stable_frames", 3);
  bc.algo.fire_cooldown = fs.getd("fire_cooldown", 0.1);
  bc.algo.fire_enabled = fs.getb("fire_enabled", true);
  bc.algo.aim_local_only = fs.getb("aim_local_only", true);
  bc.algo.fire_require_lock = fs.getb("fire_require_lock", true);
  bc.algo.fire_rms_gate_deg = fs.getd("fire_rms_gate_deg", 15.0);
  bc.algo.fire_min_n = fs.geti("fire_min_n", 15);
  bc.algo.visualize = fs.getb("visualize", false);
  bc.algo.log_every = fs.geti("log_every", 30);
  bc.algo.console_every = fs.geti("console_every", bc.algo.console_every);
  bc.algo.visualize_every = fs.geti("visualize_every", bc.algo.visualize_every);
  bc.algo.log_path = fs.gets("log_path", "hw26_bridge_log.csv");
  {
    const std::string fr = fs.gets("frame", "face");
    bc.algo.frame = (fr == "gimbal")    ? hw26_bridge::RefFrame::kGimbal
                    : (fr == "chassis") ? hw26_bridge::RefFrame::kChassis
                                        : hw26_bridge::RefFrame::kFace;
  }
  bc.algo.face_ema = fs.getd("face_ema", bc.algo.face_ema);
  bc.algo.face_ema_locked = fs.getd("face_ema_locked", bc.algo.face_ema_locked);
  bc.algo.face_lock_frames = fs.geti("face_lock_frames", bc.algo.face_lock_frames);
  bc.algo.max_slew_deg_s = fs.getd("max_slew_deg_s", bc.algo.max_slew_deg_s);
  bc.algo.cmd_smooth_alpha = fs.getd("cmd_smooth_alpha", bc.algo.cmd_smooth_alpha);
  bc.algo.max_aim_step_px = fs.getd("max_aim_step_px", bc.algo.max_aim_step_px);
  bc.algo.stale_ms = fs.getd("stale_ms", bc.algo.stale_ms);
  bc.image_topic = fs.gets("image_topic", bc.image_topic);
  bc.pose_topic = fs.gets("pose_topic", bc.pose_topic);
  bc.cmd_topic = fs.gets("cmd_topic", bc.cmd_topic);
  bc.max_image_age_ms = fs.getd("max_image_age_ms", 300.0);
  // 与 ④ 的 stale 门统一：取较小者，避免两个不一致的过期判据
  bc.max_image_age_ms = std::min(bc.max_image_age_ms, bc.algo.stale_ms);
  return bc;
}

