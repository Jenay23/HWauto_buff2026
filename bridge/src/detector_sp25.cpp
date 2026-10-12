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
constexpr int kTopK = 64;      // 只看前 64 个候选就够找第二片了
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
std::vector<Sp25Detection> Sp25Detector::detect_multi(const cv::Mat &bgr, float thr1, float thr2,
                                                      float ratio2, float min_sep_px) {
  float inv_scale = 1.0f;
  const float *out = run(bgr, &inv_scale);

  std::vector<std::pair<float, int>> cand;
  cand.reserve(kOutCols);
  for (int i = 0; i < kOutCols; ++i) cand.emplace_back(out[4 * kOutCols + i], i);
  const std::size_t kk = std::min<std::size_t>(kTopK, cand.size());
  std::partial_sort(cand.begin(), cand.begin() + kk, cand.end(),
                    [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                      return a.first > b.first;
                    });

  std::vector<Sp25Detection> res;
  if (cand.empty() || cand[0].first < thr1) return res;

  Sp25Detection d1 = decode_column(out, inv_scale, cand[0].second);
  d1.max_score = cand[0].first;
  const cv::Point2f c1(d1.box.x + d1.box.width * 0.5f, d1.box.y + d1.box.height * 0.5f);
  res.push_back(d1);

  const float need = std::max(thr2, ratio2 * cand[0].first);
  for (std::size_t k = 1; k < kk; ++k) {
    if (cand[k].first < need) break;  // 已排序 -> 后面只会更小
    Sp25Detection d2 = decode_column(out, inv_scale, cand[k].second);
    d2.max_score = cand[0].first;
    const cv::Point2f c2(d2.box.x + d2.box.width * 0.5f, d2.box.y + d2.box.height * 0.5f);
    if (cv::norm(c2 - c1) < min_sep_px) continue;  // 同一片靶板的重叠框 -> 跳过
    res.push_back(d2);
    break;
  }
  return res;
}

}  // namespace hw26_bridge
