#include "hw26_pipeline.hpp"

#include <cmath>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

namespace hw26_bridge {

// 从 hw26_pipeline.cpp 原样搬出（P2-2），逻辑一行未改。
// 声明仍在 hw26_pipeline.hpp，所以调用点不需要动。

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

}  // namespace hw26_bridge
