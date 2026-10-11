#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预测器对比（只用日志，不需要模拟器）。
    python3 tools/predictor_compare.py [日志...]
不指定日志则取 logs/ 里最新的几份。

从 [bridge] 逐帧行里取 有效观测(det>0 且 solved=1) 的 roll 序列，
按断帧切成"点亮窗口"，然后在每个窗口内：
    用最后 N 个点拟合某个模型 -> 向前预测 h 步 -> 与真实值比 -> 统计误差(度)
比较这些预测器：
    const   : 保持最后一个值
    lin     : 一次外推
    quad    : 二次外推
    ar(p)   : AR(p) 自回归，多步递推（p=1..6）
    sine    : c0 + c1*t + c2*sin(w t) + c3*cos(w t)，w 网格搜索
    csine   : 官方约束模型 theta0 + sgn*[(2.090-a)t + p(1-cos wt) + q sin wt]，(a,w) 网格
            a in [0.780,1.045], w in [1.884,2.000]   <- 用上了模拟器规则的硬边界
另外做一次"速率一致性检查"：我们的 roll 速率是否落在模型包络内（<=2.090 rad/s）。
"""
import math
import re
import sys
from pathlib import Path

PI = math.pi
DEG = 180.0 / PI


def wrap(a):
    while a > PI:
        a -= 2 * PI
    while a <= -PI:
        a += 2 * PI
    return a


BRIDGE_RE = re.compile(
    r"\[bridge\]\s+#(?P<k>\d+)\s+age=(?P<age>[-\d.]+)ms\s+det=(?P<det>\d+)\s+"
    r"sel=(?P<sel>\d+)\s+solved=(?P<sol>\d+).*?st=(?P<st>\d+)\s+dis=(?P<dis>[-\d.]+)\s+"
    r"roll=(?P<roll>[-\d.]+)\s+w=(?P<w>[-\d.]+).*?lat=(?P<lat>[-\d.]+)ms"
)


def parse_log(path):
    """-> [(frame_index, roll, lat_ms, det, solved, st), ...] 逐帧"""
    out = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = BRIDGE_RE.search(line)
        if not m:
            continue
        out.append(
            (
                int(m.group("k")),
                float(m.group("roll")),
                float(m.group("lat")),
                int(m.group("det")),
                int(m.group("sol")),
                int(m.group("st")),
            )
        )
    return out


def valid_windows(frames, max_gap=3):
    """取 det>0 且 solved=1 的帧，按帧号间隔切窗口，返回 [[...], ...]"""
    sel = [f for f in frames if f[3] > 0 and f[4] == 1]
    wins, cur, prev_k = [], [], None
    for f in sel:
        if prev_k is not None and f[0] - prev_k > max_gap:
            if len(cur) >= 8:
                wins.append(cur)
            cur = []
        cur.append(f)
        prev_k = f[0]
    if len(cur) >= 8:
        wins.append(cur)
    return wins


# ---------------- 预测器 ----------------
def fit_const(ts, ys):
    return lambda th: ys[-1]


def fit_lin(ts, ys):
    t0 = ts[-1]
    xs = [t - t0 for t in ts]
    n = len(xs)
    sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return lambda th: ys[-1]
    a1 = (n * sxy - sx * sy) / den
    a0 = (sy - a1 * sx) / n
    return lambda h: a0 + a1 * (h * DT)


def fit_quad(ts, ys):
    t0 = ts[-1]
    xs = [t - t0 for t in ts]
    n = len(xs)
    S = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
    b = [sum((x ** i) * y for x, y in zip(xs, ys)) for i in range(3)]
    # 3x3 高斯消元（带小 ridge）
    for i in range(3):
        S[i][i] += 1e-9
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(S[r][i]))
        S[i], S[p] = S[p], S[i]
        b[i], b[p] = b[p], b[i]
        if abs(S[i][i]) < 1e-14:
            return lambda th: ys[-1]
        for r in range(i + 1, 3):
            f = S[r][i] / S[i][i]
            for c in range(i, 3):
                S[r][c] -= f * S[i][c]
            b[r] -= f * b[i]
    c = [0.0] * 3
    for i in (2, 1, 0):
        c[i] = (b[i] - sum(S[i][j] * c[j] for j in range(i + 1, 3))) / S[i][i]
    return lambda h: c[0] + c[1] * (h * DT) + c[2] * (h * DT) ** 2


def fit_ar(ts, ys, p):
    """theta(k) = sum_i c_i theta(k-i)，最小二乘；多步预测靠递推"""
    if len(ys) <= p + 2:
        return None
    rows, rhs = [], []
    for k in range(p, len(ys)):
        rows.append([ys[k - i - 1] for i in range(p)])
        rhs.append(ys[k])
    n = len(rows)
    A = [[sum(r[i] * r[j] for r in rows) for j in range(p)] for i in range(p)]
    b = [sum(r[i] * y for r, y in zip(rows, rhs)) for i in range(p)]
    for i in range(p):
        A[i][i] += 1e-9
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        if abs(A[i][i]) < 1e-14:
            return None
        for r in range(i + 1, p):
            f = A[r][i] / A[i][i]
            for c in range(i, p):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    c = [0.0] * p
    for i in range(p - 1, -1, -1):
        c[i] = (b[i] - sum(A[i][j] * c[j] for j in range(i + 1, p))) / A[i][i]
    hist = list(ys)

    def pred(h):
        hh = list(hist)
        for _ in range(h):
            hh.append(sum(c[i] * hh[-i - 1] for i in range(p)))
        return hh[-1]

    return pred


def _solve(M, v):
    n = len(v)
    M = [row[:] for row in M]
    v = v[:]
    for i in range(n):
        M[i][i] += 1e-9
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        v[i], v[piv] = v[piv], v[i]
        if abs(M[i][i]) < 1e-14:
            return None
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n):
                M[r][c] -= f * M[i][c]
            v[r] -= f * v[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (v[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _lsq(basis, ys):
    """basis: [[f1..fm]...] -> 返回系数 或 None"""
    m = len(basis[0])
    A = [[sum(b[i] * b[j] for b in basis) for j in range(m)] for i in range(m)]
    v = [sum(b[i] * y for b, y in zip(basis, ys)) for i in range(m)]
    return _solve(A, v)


def fit_sine(ts, ys, nw=48):
    t0 = ts[-1]
    xs = [t - t0 for t in ts]
    best = None
    for gi in range(nw + 1):
        w = 1.5 + (2.5 - 1.5) * gi / nw
        basis = [[1.0, x, math.sin(w * x), math.cos(w * x)] for x in xs]
        c = _lsq(basis, ys)
        if not c:
            continue
        se = sum((y - (c[0] + c[1] * x + c[2] * math.sin(w * x) + c[3] * math.cos(w * x))) ** 2
                 for x, y in zip(xs, ys))
        if best is None or se < best[0]:
            best = (se, c, w)

    def pred(h):
        if best is None:
            return ys[-1]
        _, c, w = best
        x = (ts[-1] + h * DT) - t0
        return c[0] + c[1] * x + c[2] * math.sin(w * x) + c[3] * math.cos(w * x)

    return pred


def fit_csine(ts, ys, na=16, nw=10):
    """官方约束: theta0 + sgn*[(2.090-a)t + p(1-cos wt) + q sin wt]"""
    t0 = ts[-1]
    xs = [t - t0 for t in ts]
    coarse = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if len(xs) > 1 and abs(xs[-1] - xs[0]) > 1e-9 else 0.0
    sgn = -1.0 if coarse < 0 else 1.0
    best = None
    for ia in range(na + 1):
        a = 0.780 + (1.045 - 0.780) * ia / na
        b = 2.090 - a
        for iw in range(nw + 1):
            w = 1.884 + (2.000 - 1.884) * iw / nw
            yv = [y - sgn * b * x for x, y in zip(xs, ys)]
            basis = [[1.0, sgn * (1 - math.cos(w * x)), sgn * math.sin(w * x)] for x in xs]
            c = _lsq(basis, yv)
            if not c:
                continue
            se = 0.0
            for x, y in zip(xs, ys):
                p = sgn * b * x + c[0] + sgn * c[1] * (1 - math.cos(w * x)) + sgn * c[2] * math.sin(w * x)
                se += (y - p) ** 2
            if best is None or se < best[0]:
                best = (se, c, a, b, w, sgn)

    def pred(h):
        if best is None:
            return ys[-1]
        _, c, a, b, w, sgn = best
        x = (ts[-1] + h * DT) - t0
        return sgn * b * x + c[0] + sgn * c[1] * (1 - math.cos(w * x)) + sgn * c[2] * math.sin(w * x)

    return pred


DT = 1.0 / 30.0   # 逐帧间隔，稍后按日志实测覆盖




# ===========================================================================
#  EKF（误差状态式，手写 4x4，不依赖 numpy）
#    状态 x = [theta, a, w, phi]
#    过程  theta' = a*sin(w*tau + phi) + (2.090 - a)     <- 官方约束 b=2.090-a
#    观测  z = theta_measured                            <- H = [1,0,0,0]
#    tau = 滤波器自己的经过时间；激活时刻未知这件事由自由 phi 吸收
#    硬约束：每步把 a / w 夹回官方区间 [0.780,1.045] / [1.884,2.000]
#    创新门：|z - theta_pred| > k*sqrt(S) -> 拒这一帧（并计数）
# ===========================================================================
class EKF:
    def __init__(self, theta0, dt, R=1e-4, q_theta=5e-3, q_a=1e-5, q_w=1e-6, q_phi=1e-5,
                 gate_k=3.0, reset_after=12, s_floor=4e-3):
        self.dt = dt
        self.x = [theta0, 0.9, 1.94, 0.0]
        self.P = [[4e-3 if i == j else 0.0 for j in range(4)] for i in range(4)]
        self.P[1][1] = 2e-2      # a 的不确定度 ~0.14
        self.P[2][2] = 3e-3      # w 的不确定度 ~0.05
        self.P[3][3] = 3.0       # phi 几乎未知（自由相位）
        self.R = R
        self.q_theta, self.q_a, self.q_w, self.q_phi = q_theta, q_a, q_w, q_phi
        self.gate_k = gate_k
        self.s_floor = s_floor
        self.reset_after = reset_after
        self.tau = 0.0
        self.rej = 0
        self.acc = 0
        self.total = 0
        self.miss = 0

    # ---- 内部：模型与雅可比 ----
    def _speed(self, x, tau):
        return x[1] * math.sin(x[2] * tau + x[3]) + (2.090 - x[1])

    def _dtheta_d(self, x, tau):
        """d speed / d[a, w, phi]"""
        s = math.sin(x[2] * tau + x[3])
        c = math.cos(x[2] * tau + x[3])
        return [s - 1.0, x[1] * tau * c, x[1] * c]

    def append(self, x, d):
        """把状态沿当前模型走一小步 dt"""
        sp = self._speed(x, self.tau)
        g = self._dtheta_d(x, self.tau)
        x2 = list(x)
        x2[0] = x[0] + sp * d
        x2[1] = x[1] + self.q_a * 0  # 常数
        x2[2] = x[2]
        x2[3] = x[3]
        return x2, g

    def predict(self, h):
        """不动状态地预测 h 帧之后的 theta"""
        x = list(self.x)
        tau = self.tau
        for _ in range(h):
            sp = self._speed(x, tau)
            x[0] = x[0] + sp * self.dt
            tau += self.dt
        return x[0]

    def step(self, z, t):
        """预测 + 带门限的更新。t 为绝对时间（用于推进 tau）"""
        # ---- 预测 ----
        dt = self.dt
        xp = list(self.x)
        sp = self._speed(self.x, self.tau)
        g = self._dtheta_d(self.x, self.tau)
        xp[0] = self.x[0] + sp * dt
        # 雅可比 F = I + A*dt，A 只有第一行非零
        F = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        F[0][1] = g[0] * dt
        F[0][2] = g[1] * dt
        F[0][3] = g[2] * dt
        Q = [[0.0] * 4 for _ in range(4)]
        Q[0][0] = self.q_theta * dt
        Q[1][1] = self.q_a
        Q[2][2] = self.q_w
        Q[3][3] = self.q_phi
        P = [[sum(F[i][k] * self.P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        P = [[sum(P[i][k] * F[j][k] for k in range(4)) for j in range(4)] for i in range(4)]
        for i in range(4):
            P[i][i] += Q[i][i]

        # ---- 门限 ----
        S = P[0][0] + self.R + self.s_floor   # 地板：门限按"测量噪声"而不是状态不确定度
        nu = wrap(z - xp[0])
        self.total += 1
        if abs(nu) > self.gate_k * math.sqrt(S + 1e-18):
            self.rej += 1
            self.miss += 1
            self.tau += dt
            if self.miss >= self.reset_after:      # 连续拒太多 -> 重初始化
                self.x = [z, self.x[1], self.x[2], self.x[3]]
                self.P = [[4e-3 if i == j else 0.0 for j in range(4)] for i in range(4)]
                self.P[1][1] = 2e-2
                self.P[2][2] = 3e-3
                self.P[3][3] = 3.0
                self.miss = 0
            return xp[0]

        # ---- 更新 ----
        self.miss = 0
        K = [P[i][0] / S for i in range(4)]
        x_new = [xp[i] + K[i] * nu for i in range(4)]
        # P <- (I-KH)P
        Pn = [[P[i][j] - K[i] * P[0][j] for j in range(4)] for i in range(4)]
        # 硬约束：a / w 夹回官方区间
        x_new[1] = min(1.045, max(0.780, x_new[1]))
        x_new[2] = min(2.000, max(1.884, x_new[2]))
        self.x = x_new
        self.P = Pn
        self.tau += dt
        return x_new[0]


def ekf_errors(ts, ys, horizons, gate_k=3.0, inject_jumps=False, seed=0):
    """对一条窗口做一次顺序 EKF；返回 {h: [误差度]} 和拒帧统计"""
    import random
    rng = random.Random(seed)
    ekf = EKF(ys[0], DT, gate_k=gate_k)
    out = {h: [] for h in horizons}
    zs = list(ys)
    if inject_jumps:
        # 人为注入跳变：约 25% 的帧，加 +-25~120 度
        for i in range(1, len(zs) - 1):
            if rng.random() < 0.25:
                zs[i] = zs[i] + rng.choice([-1, 1]) * rng.uniform(25, 120) / DEG
    for i in range(len(zs)):
        for h in horizons:
            if i + h < len(zs):
                out[h].append(abs(wrap(ekf.predict(h) - zs[i + h])) * DEG)
        ekf.step(zs[i], i * DT)
    return out, ekf.rej, ekf.total


def run_ekf_block(clean, horizons, title, inject_jumps):
    """对给定窗口集合跑 EKF，打印表格"""
    print(f"\n  --- EKF（{title}）---")
    for h in horizons:
        allv, cs = [], []
        rej = tot = 0
        for idx, (pth, w, dt, ok) in enumerate(clean):
            ts = [f[0] * DT for f in w]
            ys, acc, prev = [], 0.0, None
            for f in w:
                if prev is not None:
                    acc += wrap(f[1] - prev)
                prev = f[1]
                ys.append(acc)
            e, r, t = ekf_errors(ts, ys, [h], inject_jumps=inject_jumps, seed=idx)
            allv += e[h]
            rej += r
            tot += t
            if ok:
                cs += e[h]
        if not allv:
            continue
        allv.sort(); cs.sort()
        med = allv[len(allv) // 2]
        p90 = allv[min(len(allv) - 1, int(0.9 * len(allv)))]
        extra = ""
        if len(cs) >= 5:
            m2 = cs[len(cs) // 2]
            p2 = cs[min(len(cs) - 1, int(0.9 * len(cs)))]
            extra = f"  | 干净: 中位 {m2:6.2f}  p90 {p2:6.2f} (n={len(cs)})"
        print(f"    ekf 预测 {h} 帧 ({h*DT*1000:.0f} ms): 全部 中位 {med:6.2f}  p90 {p90:6.2f} "
              f"(n={len(allv)}){extra}   拒帧 {rej}/{tot} = {100.0*rej/max(1,tot):.0f}%")


def main():
    global DT
    paths = sys.argv[1:]
    if not paths:
        logs = sorted(Path("logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        paths = [str(p) for p in logs[:6]]
    print(f"用 {len(paths)} 份日志: " + ", ".join(Path(p).name for p in paths))

    all_win = []
    lat_all, rate_all = [], []
    for p in paths:
        fr = parse_log(p)
        if len(fr) < 10:
            continue
        lat_all += [f[2] for f in fr]
        wins = valid_windows(fr)
        # 用有效帧的 lat 中位数估计 dt
        lats = [f[2] for w in wins for f in w]
        if lats:
            dt = sorted(lats)[len(lats) // 2] / 1000.0
            all_win.append((p, wins, dt))
            # 速率一致性：窗口内相邻帧的 wrap 差 / dt
            for w in wins:
                for i in range(1, len(w)):
                    d = wrap(w[i][1] - w[i - 1][1]) / dt
                    rate_all.append(d)

    if not all_win:
        print("没有可用窗口"); return
    DTs = [dt for _, _, dt in all_win]
    DT = sorted(DTs)[len(DTs) // 2]
    print(f"估计逐帧间隔 dt = {DT*1000:.1f} ms（由 lat 中位数）；窗口数 = {sum(len(w) for _, w, _ in all_win)}")

    # ---- 速率一致性检查 ----
    if rate_all:
        rate_all.sort()
        n = len(rate_all)
        q = lambda f: rate_all[min(n - 1, int(f * n))]
        inside = sum(1 for r in rate_all if abs(r) <= 2.090)
        print("\n=== 速率一致性检查（模型包络 |speed| <= 2.090 rad/s）===")
        print(f"  样本 {n}；|rate|<=2.090 的比例: {100.0*inside/n:.1f}%")
        print(f"  rate 分位: p05={q(.05):.2f} p25={q(.25):.2f} p50={q(.50):.2f} "
              f"p75={q(.75):.2f} p95={q(.95):.2f} rad/s")
        print(f"  模型允许的平均速度区间: [1.045, 1.310] rad/s")

    # 标记"干净窗口"：窗口内没有任何一帧的 |rate| 超过物理上限的 1.2 倍
    clean = []
    for pth, wins, dt in all_win:
        for w in wins:
            ok = True
            for i in range(1, len(w)):
                if abs(wrap(w[i][1] - w[i - 1][1]) / dt) > 2.090 * 1.2:
                    ok = False
                    break
            clean.append((pth, w, dt, ok))
    n_clean = sum(1 for *_x, ok in clean if ok)
    print(f"\n=== 干净窗口（窗口内没有超过物理速率上限的帧）: {n_clean}/{len(clean)} "
          f"= {100.0*n_clean/max(1,len(clean)):.0f}% ===")

    # ---- 预测器对比 ----
    HOR = [3, 6, 9, 12]          # 帧 -> 约 0.1/0.2/0.3/0.4 s
    NWIN = [15, 20]              # 拟合窗口长度
    names = ["const", "lin", "quad", "ar2", "ar4", "ar6", "sine", "csine"]

    print("\n=== 多步预测误差（|wrap(预测-真值)| 的中位数 / p90，单位: 度）===")
    # ---- EKF 两段：真实数据 + 模拟跳变 ----
    run_ekf_block(clean, HOR, "真实数据（含日志里原有的跳变）", inject_jumps=False)
    run_ekf_block(clean, HOR, "模拟跳变（人为注入 25% 的 25~120 度跳变）", inject_jumps=True)

    for N in NWIN:
        for H in HOR:
            res = {k: [] for k in names}
            resC = {k: [] for k in names}
            for pth, w, dt, ok in clean:
                    if len(w) < N + H + 1:
                        continue
                    ts = [f[0] * DT for f in w]        # 用帧号*dt 当时间轴（单调、等间隔）
                    ys = []
                    acc = 0.0
                    prev = None
                    for f in w:                        # 逐窗口解缠
                        if prev is not None:
                            acc += wrap(f[1] - prev)
                        prev = f[1]
                        ys.append(acc)
                    fit_ts, fit_ys = ts[: N], ys[: N]
                    # 目标：把窗口向后滚动，用前 N 点预测第 N..N+H
                    for s in range(0, len(w) - N - H):
                        fts, fys = ts[s:s + N], ys[s:s + N]
                        tgt = ys[s + N + H - 1]
                        preds = {
                            "const": fit_const(fts, fys),
                            "lin": fit_lin(fts, fys),
                            "quad": fit_quad(fts, fys),
                            "sine": fit_sine(fts, fys),
                            "csine": fit_csine(fts, fys),
                        }
                        for p in (2, 4, 6):
                            fn = fit_ar(fts, fys, p)
                            if fn:
                                preds[f"ar{p}"] = (lambda fn_: (lambda h: fn_(h)))(fn)
                        for k, fn in preds.items():
                            try:
                                pv = fn(H)
                            except Exception:
                                continue
                            e = abs(wrap(pv - tgt)) * DEG
                            res[k].append(e)
                            if ok:
                                resC[k].append(e)
            print(f"\n  窗口 N={N} 帧, 预测 {H} 帧 ({H*DT*1000:.0f} ms):")
            rows = []
            for k in names:
                v = sorted(res[k])
                if len(v) < 5:
                    continue
                med = v[len(v) // 2]
                p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
                rows.append((med, p90, k, len(v)))
            rows.sort()
            for med, p90, k, cnt in rows:
                v2 = sorted(resC[k])
                extra = ""
                if len(v2) >= 5:
                    m2 = v2[len(v2) // 2]
                    p2 = v2[min(len(v2) - 1, int(0.9 * len(v2)))]
                    extra = f"   | 干净: 中位 {m2:6.2f}  p90 {p2:6.2f} (n={len(v2)})"
                print(f"    {k:>6}: 全部 中位 {med:6.2f}  p90 {p90:6.2f} (n={cnt}){extra}")


if __name__ == "__main__":
    main()
