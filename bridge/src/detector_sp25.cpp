#include "detector_sp25.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace hw26_bridge {

namespace {
constexpr int kOutRows = 17;   // 4 bbox + 1 score + 6*2 kpt
constexpr int kOutCols = 8400;
constexpr int kNumPoints = 6;
}  // namespace

Sp25Detector::Sp25Detector(const std::string &model_path) {
  auto model = core_.read_model(model_path);
  const ov::Shape shp = model->input().get_shape();
  if (shp.size() != 4 || shp[0] != 1 || shp[1] != 3) {
    throw std::runtime_error("sp25 detector: input must be [1,3,H,W]");
  }
  in_size_ = cv::Size(static_cast<int>(shp[3]), static_cast<int>(shp[2]));
  compiled_ = core_.compile_model(model, "CPU");
  req_ = compiled_.create_infer_request();

  const ov::Shape oshp = model->output().get_shape();
  if (oshp.size() != 3 || static_cast<int>(oshp[1]) != kOutRows ||
      static_cast<int>(oshp[2]) != kOutCols) {
    throw std::runtime_error("sp25 detector: unexpected output shape");
  }
}

// 预处理 + 推理：严格照抄 sp25 的 fill_tensor_data_image（等比缩放、贴左上角、其余填黑），
// 然后 BGR u8 -> RGB f32/255 -> CHW。返回输出张量指针，inv_scale 用于把坐标还原到原图。
const float *Sp25Detector::run(const cv::Mat &bgr, float *inv_scale) {
  const float scale = std::min(static_cast<float>(in_size_.height) / bgr.rows,
                               static_cast<float>(in_size_.width) / bgr.cols);
  const cv::Matx23f m{scale, 0.f, 0.f, 0.f, scale, 0.f};
  cv::Mat blob;
  cv::warpAffine(bgr, blob, m, in_size_);  // 贴左上角，其余填 0（黑边）

  cv::Mat f32;
  blob.convertTo(f32, CV_32F, 1.0 / 255.0);
  cv::cvtColor(f32, f32, cv::COLOR_BGR2RGB);
  const int H = in_size_.height, W = in_size_.width;
  ov::Tensor in_t(compiled_.input().get_element_type(), compiled_.input().get_shape());
  float *dst = in_t.data<float>();
  std::vector<cv::Mat> ch(3);
  cv::split(f32, ch);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(dst + static_cast<size_t>(c) * H * W, ch[c].ptr<float>(),
                sizeof(float) * H * W);
  }
  req_.set_input_tensor(in_t);
  req_.infer();
  *inv_scale = 1.0f / scale;
  return req_.get_output_tensor(0).data<const float>();
}

Sp25Detection Sp25Detector::decode_column(const float *out, float inv_scale, int col) const {
  Sp25Detection d;
  d.confidence = out[4 * kOutCols + col];
  d.max_score = d.confidence;
  for (int k = 0; k < kNumPoints; ++k) {
    const float x = out[(5 + k * 2) * kOutCols + col] * inv_scale;
    const float y = out[(5 + k * 2 + 1) * kOutCols + col] * inv_scale;
    d.kpt[static_cast<std::size_t>(k)] = cv::Point2f(x, y);
  }
  // R 中心粗估：sp25 的 buff_detector.cpp:42-43
  d.r_center = d.kpt[4] + 1.4f * (d.kpt[5] - d.kpt[4]);
  const float cx = out[0 * kOutCols + col] * inv_scale;
  const float cy = out[1 * kOutCols + col] * inv_scale;
  const float w = out[2 * kOutCols + col] * inv_scale;
  const float h = out[3 * kOutCols + col] * inv_scale;
  d.box = cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
  return d;
}

// 原行为：只取全局最高分那一个
std::optional<Sp25Detection> Sp25Detector::detect(const cv::Mat &bgr, float conf_threshold) {
  float inv_scale = 1.0f;
  const float *out = run(bgr, &inv_scale);
  int best = -1;
  float best_score = 0.0f;
  for (int i = 0; i < kOutCols; ++i) {
    const float s = out[4 * kOutCols + i];
    if (s > best_score) { best_score = s; best = i; }
  }
  if (best < 0 || best_score < conf_threshold) return std::nullopt;
  Sp25Detection d = decode_column(out, inv_scale, best);
  d.max_score = best_score;
  return d;
}

// 大符双靶标：允许置信度最高的两个，但第二片必须同时过
//   ① 绝对阈值 thr2      ② 相对阈值 ratio2*最高分      ③ 与第一片中心分离 min_sep_px
// 不满足就只返回 1 个（自动回落成单片）
std::vector<Sp25Detection> Sp25Detector::detect_multi(const cv::Mat &bgr, float thr) {
  float inv_scale = 1.0f;
  const float *out = run(bgr, &inv_scale);

  // 收集所有 score > thr 的候选
  std::vector<std::pair<float, int>> hits;
  for (int i = 0; i < kOutCols; ++i) {
    const float sc = out[4 * kOutCols + i];
    if (sc > thr) hits.emplace_back(sc, i);
  }
  std::vector<Sp25Detection> res;
  if (hits.empty()) return res;
  const std::size_t kMaxOut = 4;   // 正常就是两片；同片重叠框刷屏时封顶
  if (hits.size() > kMaxOut) {
    std::partial_sort(hits.begin(), hits.begin() + kMaxOut, hits.end(),
                      [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                        return a.first > b.first;
                      });
    hits.resize(kMaxOut);
  }
  for (const auto &h : hits) {
    Sp25Detection d = decode_column(out, inv_scale, h.second);
    d.max_score = h.first;
    res.push_back(d);
  }
  return res;
}

}  // namespace hw26_bridge
