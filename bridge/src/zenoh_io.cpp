#include "zenoh_io.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "app_state.hpp"

// 本文件 = "字节 <-> 结构体" 的那一层：解析、拼装、四个订阅回调、--capture 落盘。
// 共享状态（g_frame/g_pose/...）在这里定义，头文件里 extern 给主循环用。

// ---------------- JSON 小工具（自己拼，避免引第三方） ----------------

bool parse_pose(const uint8_t *d, size_t n, GimbalPose *out) {
  std::string s(reinterpret_cast<const char *>(d), n);
  auto grab = [&](const char *key, float *dst) -> bool {
    const std::string k = std::string("\"") + key + "\"";
    const size_t p = s.find(k);
    if (p == std::string::npos) return false;
    const size_t c = s.find(':', p + k.size());
    if (c == std::string::npos) return false;
    *dst = std::strtof(s.c_str() + c + 1, nullptr);
    return true;
  };
  bool any = false;
  any |= grab("yaw", &out->yaw);
  any |= grab("pitch", &out->pitch);
  any |= grab("roll", &out->roll);
  return any;
}

std::string make_cmd_json(double yaw, double pitch, double roll, bool fire, bool control) {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "{\"yaw\":%.6f,\"pitch\":%.6f,\"roll\":%.6f,\"fire\":%s,\"control\":%s}", yaw,
                pitch, roll, fire ? "true" : "false", control ? "true" : "false");
  return std::string(buf);
}

// ---------------- BIMG 解析 ----------------

bool parse_bimg(const uint8_t *d, size_t n, BimgFrame *f) {
  if (n < 20 || std::memcmp(d, "BIMG", 4) != 0) return false;
  std::memcpy(&f->timestamp_ns, d + 4, 8);
  std::memcpy(&f->w, d + 12, 4);
  std::memcpy(&f->h, d + 16, 4);
  f->rgb = d + 20;
  f->rgb_len = n - 20;
  return f->rgb_len == static_cast<size_t>(f->w) * f->h * 3;
}

// ---------------- 最新帧邮箱 ----------------

LatestFrame g_frame;
std::mutex g_pose_mu;
GimbalPose g_pose;
std::atomic<bool> g_have_pose{false};
std::atomic<uint64_t> g_pose_count{0};
std::atomic<uint64_t> g_img_count{0};
std::atomic<uint64_t> g_dropped{0};
std::string g_capture_path;

// ---- rt/rune_kpts：模拟器发布的"每个靶标 8 个真值点"（JSON），只留最新一帧写日志用 ----
std::mutex g_kpts_mu;
std::string g_kpts_json;
std::atomic<uint64_t> g_kpts_count{0};
std::ofstream g_kpts_log;

void on_rune_kpts(struct z_loaned_sample_t *sample, void *) {
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const char *d = reinterpret_cast<const char *>(z_slice_data(z_loan(owned)));
  const size_t n = z_slice_len(z_loan(owned));
  {
    std::lock_guard<std::mutex> lk(g_kpts_mu);
    g_kpts_json.assign(d, n);
    g_kpts_count++;
  }
  z_drop(z_move(owned));
}

// ---- rt/stat：模拟器发布的命中统计 {launch, accurate, pct}（5Hz，只读）----
//   launch   总发弹数（键盘 Space 与 cmd/gimbal.fire 都计入）
//   accurate 在激活窗口内命中符的弹数；pct = accurate/launch
//   用途：把"拟合好坏"和【真实命中率】挂钩 —— 这是唯一能判断坏窗值不值得继续修的指标。
std::mutex g_stat_mu;
std::string g_stat_path;                 // 由 worker 打开 kpts 日志时设置
std::atomic<uint64_t> g_stat_count{0};
std::mutex g_stat_fp_mu;
std::ofstream g_stat_log;
bool g_stat_tried = false;
long long g_stat_launch = 0, g_stat_accurate = 0, g_stat_hits = 0;
double g_stat_pct = 0.0;

// 极简取值：payload 是 serde_json 的固定格式 {"launch":N,"accurate":N,"pct":F}
static bool json_num(const char *s, const char *key, double *out) {
  const char *p = std::strstr(s, key);
  if (p == nullptr) return false;
  p += std::strlen(key);
  while (*p == ' ' || *p == ':' || *p == '"') ++p;
  char *end = nullptr;
  const double v = std::strtod(p, &end);
  if (end == p) return false;
  *out = v;
  return true;
}

void on_rune_stat(struct z_loaned_sample_t *sample, void *) {
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const char *d = reinterpret_cast<const char *>(z_slice_data(z_loan(owned)));
  const size_t n = z_slice_len(z_loan(owned));
  std::string js(d, n);
  double lc = 0, ac = 0, pc = 0, hc = 0;
  json_num(js.c_str(), "\"launch\"", &lc);
  json_num(js.c_str(), "\"accurate\"", &ac);
  json_num(js.c_str(), "\"pct\"", &pc);
  json_num(js.c_str(), "\"hits\"", &hc);   // ★ 真命中（HUD 上那个 hits=）
  {
    std::lock_guard<std::mutex> lk(g_stat_mu);
    g_stat_launch = static_cast<long long>(lc);
    g_stat_accurate = static_cast<long long>(ac);
    g_stat_hits = static_cast<long long>(hc);
    g_stat_pct = pc;
    g_stat_count++;
    std::lock_guard<std::mutex> lk2(g_stat_fp_mu);
    if (!g_stat_tried && !g_stat_path.empty()) {
      g_stat_tried = true;
      g_stat_log.open(g_stat_path, std::ios::trunc);
      if (g_stat_log) {
        // launch=发弹数, hits=真命中(打中点亮片), accurate=接触次数(可>launch)
        g_stat_log << "wall_s,launch,hits,accurate,pct\n";
        std::cout << "[bridge] rt/stat log -> " << g_stat_path << std::endl;
      }
    }
    if (g_stat_log.is_open()) {
      const double ws = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count() / 1000.0;
      g_stat_log << std::fixed << std::setprecision(3) << ws << ',' << g_stat_launch << ','
                 << g_stat_hits << ',' << g_stat_accurate << ',' << std::setprecision(4)
                 << g_stat_pct << '\n';
      g_stat_log.flush();
    }
  }
  z_drop(z_move(owned));
}

void on_pose(struct z_loaned_sample_t *sample, void *) {
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const uint8_t *d = z_slice_data(z_loan(owned));
  const size_t n = z_slice_len(z_loan(owned));
  GimbalPose p;
  if (parse_pose(d, n, &p)) {
    std::lock_guard<std::mutex> lk(g_pose_mu);
    g_pose = p;
    g_have_pose = true;
    g_pose_count++;
  }
  z_drop(z_move(owned));
}

void on_image(struct z_loaned_sample_t *sample, void *) {
  const struct z_loaned_bytes_t *pl = z_sample_payload(sample);
  struct z_owned_slice_t owned;
  if (z_bytes_to_slice(pl, &owned) != Z_OK) return;
  const uint8_t *d = z_slice_data(z_loan(owned));
  const size_t n = z_slice_len(z_loan(owned));

  if (!g_capture_path.empty()) {
    BimgFrame f;
    if (parse_bimg(d, n, &f)) {
      cv::Mat rgb(f.h, f.w, CV_8UC3, const_cast<uint8_t *>(f.rgb));
      cv::Mat bgr;
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      if (cv::imwrite(g_capture_path, bgr)) {
        std::cout << "[bridge] captured " << f.w << "x" << f.h << " -> " << g_capture_path
                  << " (ts=" << f.timestamp_ns << ")" << std::endl;
      } else {
        std::cerr << "[bridge] imwrite failed: " << g_capture_path << std::endl;
      }
      g_stop = true;
    }
  } else {
    BimgFrame f;
    if (!parse_bimg(d, n, &f)) {
      std::cerr << "[bridge] bad BIMG frame (len=" << n << ")" << std::endl;
    } else {
      g_img_count++;
      g_frame.put(d, n, f.timestamp_ns);
    }
  }
  z_drop(z_move(owned));
}

int run_capture(const BridgeConfig &bc, const std::string &out_path, int timeout_s) {
  g_capture_path = out_path;
  z_owned_config_t cfg;
  z_config_default(&cfg);
  z_owned_session_t session;
  if (z_open(&session, z_move(cfg), NULL) != Z_OK) {
    std::cerr << "[bridge] z_open failed" << std::endl;
    return 1;
  }
  z_owned_closure_sample_t cb;
  z_closure_sample(&cb, on_image, NULL, NULL);
  z_view_keyexpr_t ke;
  z_view_keyexpr_from_str(&ke, bc.image_topic.c_str());
  z_owned_subscriber_t sub;
  if (z_declare_subscriber(z_loan(session), &sub, z_loan(ke), z_move(cb), NULL) != Z_OK) {
    std::cerr << "[bridge] subscribe failed: " << bc.image_topic << std::endl;
    return 2;
  }
  std::cout << "[bridge] subscribed " << bc.image_topic << ", waiting for one frame ..."
            << std::endl;
  const auto t0 = std::chrono::steady_clock::now();
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
      std::cerr << "[bridge] timeout waiting for frame (simulator running?)" << std::endl;
      return 3;
    }
  }
  return 0;
}
