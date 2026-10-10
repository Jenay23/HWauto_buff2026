// detector_sp25.hpp — 用 sp_vision_25 的 yolo11_buff 模型做检测
//
// 为什么用它：④ sp25_buff 桥已在模拟器上实测 solved≈94%、buff_dis 6~7 m 正确、
// 重投影误差 0.05~0.5 px。而 HWauto 自带的 YOLO11s-pose 模型在模拟器上
// 输出的是**满画面乱飘的 0.9 置信度误检**（背景/杂物假阳性），不可用。
//
// 这个封装严格复刻 sp25 的预处理约定（yolo11_buff.cpp:238-278）：
//   * letterbox = 等比缩放后**贴左上角**、其余填黑（不是居中灰边！）
//   * 先缩放再 /255 再 BGR->RGB，最后 HWC -> CHW
//   * 关键点坐标乘以 1/scale 还原（因为图在原点，没有 pad 偏移）
// 输出 [1,17,8400] 的语义（yolo11_buff.cpp:157-178）：
//   行 0-3 = cx,cy,w,h   行 4 = score   行 5..16 = 6 个关键点 (x,y)
//   关键点顺序（与 sp25 的 OBJECT_POINTS 配对，见 buff_solver.hpp:54-58）：
//     kpt[0]=上  kpt[1]=左  kpt[2]=下  kpt[3]=右  kpt[4]=扇叶中心  kpt[5]=臂上一点
#pragma once

#include <array>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

namespace hw26_bridge {

struct Sp25Detection {
  float confidence = 0.0f;
  std::array<cv::Point2f, 6> kpt{};  // 图像坐标（原图尺度）
  // 由 kpt[4],kpt[5] 外推的 R 中心（sp25 的做法，buff_detector.cpp:42-43）
  cv::Point2f r_center{0.f, 0.f};
  cv::Rect2f box{};   // 模型输出的 bbox（原图尺度），用于 R 中心精修的半径估计
  float max_score = 0.0f;  // 全候选最高分（不受阈值影响），用于诊断
};

class Sp25Detector {
 public:
  explicit Sp25Detector(const std::string &model_path);

  // 取置信度最高的一个框（与 sp25 的 get_onecandidatebox 一致）
  std::optional<Sp25Detection> detect(const cv::Mat &bgr, float conf_threshold);

  cv::Size input_size() const { return in_size_; }

 private:
  ov::Core core_;
  ov::CompiledModel compiled_;
  ov::InferRequest req_;
  cv::Size in_size_{640, 640};
};

}  // namespace hw26_bridge
