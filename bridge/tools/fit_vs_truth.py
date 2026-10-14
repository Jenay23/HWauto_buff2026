#!/usr/bin/env python3
"""把桥【拟合出的】参数和模拟器【真值】参数逐帧比。

用法: python3 tools/fit_vs_truth.py <hw26_bridge_log.csv> <*.kpts.jsonl>

数据来源:
  CSV : sim_t_ns, local_amp(拟合 a), local_omega(拟合 w), local_theta_rad
  jsonl: sim.true_a / sim.true_omega / sim.true_sgn （模拟器直接给的真值）
"""
import csv, json, sys, math, bisect

def load_csv(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ts = int(r["sim_t_ns"])
                a = float(r["local_amp"]); w = float(r["local_omega"])
                rows[ts] = (a, w, float(r["local_theta_rad"]))
            except Exception:
                pass
    return rows

def load_true(path):
    out = []
    for line in open(path):
        i = line.find(',')
        if i < 0: continue
        try: rec = json.loads(line[i+1:])
        except Exception: continue
        sim = rec.get("sim")
        if not sim: continue
        ta = sim.get("true_a"); tw = sim.get("true_omega")
        if ta is None or tw is None: continue
        out.append((int(line[:i]), float(ta), float(tw), float(sim.get("true_sgn") or 1)))
    out.sort()
    return out

def med(a):
    a = sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a, q):
    a = sorted(a); return a[min(len(a)-1, int(q*len(a)))] if a else float('nan')

def main(csvp, jsonlp):
    cs = load_csv(csvp); tr = load_true(jsonlp)
    if not cs or not tr:
        print("数据不足: csv=%d true=%d" % (len(cs), len(tr))); return
    tss = [x[0] for x in tr]
    da = []; dw = []; njoin = 0; nvalid = 0
    for ts, ta, tw, sg in tr:
        # CSV 用 sim_t_ns，jsonl 第一列也是图片时间戳 -> 直接找最近的
        j = bisect.bisect_left(tss, ts)
        got = None
        for k in (j-1, j, j+1):
            if 0 <= k < len(tss):
                got = tss[k]; break
        # 直接用字典按 ts 找（两边同一帧应完全相同）
        if ts in cs:
            a, w, th = cs[ts]; njoin += 1
            if a != 0.0:                      # mode7 拟合有效才比
                nvalid += 1
                da.append((a - ta)*180/math.pi if False else a - ta)
                dw.append(w - tw)
    print("真值帧 %d, 与 CSV 对齐 %d, 其中拟合有效 %d" % (len(tr), njoin, nvalid))
    if not nvalid: print("没有可比的帧（local_amp 全 0？）"); return
    print("【拟合 a - 真值 a】: 中位=%+.4f p10=%+.4f p90=%+.4f | |误差| 中位=%.4f p90=%.4f" % (
        med(da), pc(da,.1), pc(da,.9), med([abs(x) for x in da]), pc([abs(x) for x in da], .9)))
    print("【拟合 w - 真值 w】: 中位=%+.4f p10=%+.4f p90=%+.4f | |误差| 中位=%.4f p90=%.4f" % (
        med(dw), pc(dw,.1), pc(dw,.9), med([abs(x) for x in dw]), pc([abs(x) for x in dw], .9)))
    print("真值范围: a=[%.3f,%.3f]  w=[%.3f,%.3f]" % (
        min(x[1] for x in tr), max(x[1] for x in tr), min(x[2] for x in tr), max(x[2] for x in tr)))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hw26_bridge_log.csv",
         sys.argv[2] if len(sys.argv) > 2 else "")
