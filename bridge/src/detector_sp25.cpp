#include "detector_sp25.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

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

std::optional<Sp25Detection> Sp25Detector::detect(const cv::Mat &bgr, float conf_threshold) {
  // ---- 预处理：严格照抄 sp25 的 fill_tensor_data_image ----
  const float scale = std::min(static_cast<float>(in_size_.height) / bgr.rows,
                               static_cast<float>(in_size_.width) / bgr.cols);
  const cv::Matx23f m{scale, 0.f, 0.f, 0.f, scale, 0.f};
  cv::Mat blob;
  cv::warpAffine(bgr, blob, m, in_size_);  // 贴左上角，其余填 0（黑边）

  // BGR u8 -> RGB f32 /255 -> CHW
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

  const float *out = req_.get_output_tensor(0).data<const float>();
  const float inv_scale = 1.0f / scale;

  // ---- 找置信度最大的列（sp25 的 get_onecandidatebox）----
  int best = -1;
  float best_score = 0.0f;
  for (int i = 0; i < kOutCols; ++i) {
    const float s = out[4 * kOutCols + i];
    if (s > best_score) {
      best_score = s;
      best = i;
    }
  }

  Sp25Detection d;
  d.max_score = best_score;
  if (best < 0 || best_score < conf_threshold) return std::nullopt;

  d.confidence = best_score;
  for (int k = 0; k < kNumPoints; ++k) {
    const float x = out[(5 + k * 2) * kOutCols + best] * inv_scale;
    const float y = out[(5 + k * 2 + 1) * kOutCols + best] * inv_scale;
    d.kpt[k] = cv::Point2f(x, y);
  }
  // R 中心粗估：sp25 的 buff_detector.cpp:42-43
  d.r_center = d.kpt[4] + 1.4f * (d.kpt[5] - d.kpt[4]);
  // bbox（cx,cy,w,h -> 左上角 + 宽高）
  {
    const float cx = out[0 * kOutCols + best] * inv_scale;
    const float cy = out[1 * kOutCols + best] * inv_scale;
    const float w = out[2 * kOutCols + best] * inv_scale;
    const float h = out[3 * kOutCols + best] * inv_scale;
    d.box = cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
  }
  return d;
}

}  // namespace hw26_bridge
