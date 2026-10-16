#include "selftest.hpp"

#include <exception>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "hw26_pipeline.hpp"

// ---- 圆拟合自检（--selftest-center）----
// 合成"一片靶标在半径 0.7 的圆上"的位置序列，验证 fit_fixed_radius_center：
//   1) 正常：40 帧跨 ~120°、5mm/1cm 噪声 -> 圆心误差应 ~mm~cm
//   2) 混入 5 帧"面内外点"（模拟 IPPE 翻转把 t 拽出去）-> 应被外点门剔掉
//   3) 样本不足 / 弧太短 -> 应**拒绝**，而不是给一个飞掉的圆心
// 注意：拟合只用面内 (y,z)，所以沿法向（深度）的偏移天然不影响结果 ——
// 这既是优点（IPPE 翻转的深度误差进不来），也意味着外点门只管面内野值。
int run_selftest_center(const BridgeConfig &bc) {
  using Eigen::Matrix3f;
  using Eigen::Vector3f;
  // 造一个面系（与管道内 6c 的构造一致：法向 + 面内基）
  Eigen::Vector3f n(0.6f, -0.7f, 0.38f);
  n.normalize();
  Eigen::Vector3f u0 = Vector3f::UnitZ() - n * n.z();
  u0.normalize();
  const Vector3f u1 = n.cross(u0);
  Eigen::Matrix3f Rc2f;
  Rc2f.row(0) = n.transpose();
  Rc2f.row(1) = u0.transpose();
  Rc2f.row(2) = u1.transpose();
  const double r = 0.7;
  const Vector3f c_true = Vector3f(5.0f, 2.0f, 1.2f) + 0.11f * n;  // 沿法向偏一点

  // 合成历史：相位从 0 扫过 span_deg；inplane_out 指定的帧额外沿 u1 偏 0.3 m
  auto make_hist = [&](int nframe, double span_deg, double noise_m,
                       const std::vector<int> &inplane_out) {
    std::deque<std::pair<double, Vector3f>> h;
    std::mt19937 rng(1234);
    std::normal_distribution<double> nd(0.0, noise_m);
    for (int i = 0; i < nframe; ++i) {
      const double ph =
          span_deg * M_PI / 180.0 * (nframe > 1 ? static_cast<double>(i) / (nframe - 1) : 0.0);
      Vector3f t = c_true + static_cast<float>(r) * (std::cos(ph) * u0 + std::sin(ph) * u1);
      t += Vector3f(static_cast<float>(nd(rng)), static_cast<float>(nd(rng)),
                    static_cast<float>(nd(rng)));
      for (int oi : inplane_out) {
        if (oi == i) t += 0.3f * u1;
      }
      h.emplace_back(i * 0.04, t);
    }
    return h;
  };

  int bad = 0, tot = 0;
  // 拟合输出的是【符面系】圆心，所以真值要先转到同一个系里比
  const Vector3f c_true_face = Rc2f * c_true;
  auto check = [&](const char *name, const std::deque<std::pair<double, Vector3f>> &h,
                   bool expect_ok, double max_err_m, int expect_n) {
    ++tot;
    Vector3f c = Vector3f::Zero();
    int nn = -1;
    double rms = -1.0;
    const bool ok = hw26_bridge::fit_fixed_radius_center(
        h, Rc2f, r, bc.algo.center_hist_min, bc.algo.center_hist_span_m,
        bc.algo.center_hist_gate_m, h.back().first, bc.algo.center_hist_keep_s, &c, &nn, &rms);
    const double err = ok ? (c - c_true_face).norm() : -1.0;
    bool pass = (ok == expect_ok);
    if (ok && expect_ok && err > max_err_m) pass = false;
    if (ok && expect_ok && expect_n > 0 && nn != expect_n) pass = false;
    if (!pass) ++bad;
    std::printf("  %-30s ok=%d err=%8.4f m n=%3d rms=%8.4f m  %s\n", name, ok ? 1 : 0, err, nn,
                rms, pass ? "" : "  <== 不符合预期");
  };

  std::printf("\n[center-selftest] R=%.3f 真值圆心=(%.3f,%.3f,%.3f)\n", r, c_true.x(), c_true.y(),
              c_true.z());
  check("40帧/120°/5mm", make_hist(40, 120.0, 0.005, {}), true, 0.01, 40);
  check("40帧/120°/1cm", make_hist(40, 120.0, 0.010, {}), true, 0.02, 40);
  check("40帧/120°/5mm+5面内外点", make_hist(40, 120.0, 0.005, {8, 16, 24, 30, 36}), true, 0.01,
        35);
  check("8帧/40°/5mm", make_hist(8, 40.0, 0.005, {}), true, 0.03, 8);
  check("6帧(不足min_n)", make_hist(6, 120.0, 0.005, {}), false, 0.0, 0);
  check("40帧/0.5°(弧太短)", make_hist(40, 0.5, 0.005, {}), false, 0.0, 0);
  std::printf("[center-selftest] %d/%d 项符合预期\n", tot - bad, tot);
  return bad == 0 ? 0 : 6;
}


int run_detect_selftest(const BridgeConfig &bc, const std::string &png) {
  cv::Mat img = cv::imread(png, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "[bridge] cannot read image: " << png << std::endl;
    return 3;
  }
  try {
    hw26_bridge::Pipeline pipe(bc.algo);
    std::cout << "[bridge] selftest on " << png << " (" << img.cols << "x" << img.rows << ")"
              << std::endl;
    const int n = pipe.self_test_image(img);
    std::cout << "[bridge] selftest result: " << n << " detection(s)" << std::endl;
    std::cout << (n > 0 ? ">>> 模型对该图有检出" : ">>> 模型对该图无检出（可能不泛化）")
              << std::endl;
    return n > 0 ? 0 : 4;
  } catch (const std::exception &e) {
    std::cerr << "[bridge] selftest failed: " << e.what() << std::endl;
    return 5;
  }
}
