#!/usr/bin/env python3
"""把模拟器真值 8 点(rt/rune_kpts) 和桥记录的模型原始 18 点做对比。

用法: python3 tools/kpts_compare.py [hw26_rune_kpts_log.jsonl]
输出: 靶标中心误差 / 符文中心误差(keypoints[8]) / 半径比 / 等比缩放后的 8 点误差
真值点在相机光学系, 用 u=fx*X/Z+cx, v=fy*Y/Z+cy 投影(fx=fy=1738.5, cx=720, cy=540)。
"""
import json, math, sys
FX = FY = 1738.5; CX = 720.0; CY = 540.0
def proj(p):
    x, y, z = p
    return (FX*x/z+CX, FY*y/z+CY) if z > 1e-6 else None
def d(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def med(a):
    a = sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a, q):
    a = sorted(a); return a[min(len(a)-1, int(q*len(a)))] if a else float('nan')

def main(path):
    ctr=[]; rune=[]; ratio=[]; nn=[]; ordv=[]; n=0; frames=0; miss=0
    for line in open(path):
        i = line.find(',')
        if i < 0: continue
        try: rec = json.loads(line[i+1:])
        except Exception: continue
        sim = rec.get("sim"); model = rec.get("model") or []
        if not sim: continue
        frames += 1
        tcs = [proj(t["center_cam"]) for t in sim["targets"]]
        tcs = [p for p in tcs if p]
        rc = (sum(p[0] for p in tcs)/len(tcs), sum(p[1] for p in tcs)/len(tcs)) if tcs else None
        lits = [t for t in sim["targets"] if t["state"] == 1]
        if not lits: continue
        md = []
        for dd in model:
            k = dd["kpts"]; pts = [(k[2*j], k[2*j+1]) for j in range(9)]
            c = (sum(p[0] for p in pts[:8])/8, sum(p[1] for p in pts[:8])/8)
            md.append((c, pts))
        for t in lits:
            gp = [proj(p) for p in t["pts_cam"]]
            if any(g is None for g in gp): continue
            gc = (sum(p[0] for p in gp)/8, sum(p[1] for p in gp)/8)
            if not md: miss += 1; continue
            c, pts = min(md, key=lambda m: d(m[0], gc))
            n += 1
            ctr.append(d(c, gc))
            if rc: rune.append(d(pts[8], rc))
            rm = sum(d(pts[j], c) for j in range(8))/8.0
            rs = sum(d(gp[j], gc) for j in range(8))/8.0
            if rs < 1e-3: continue
            ratio.append(rm/rs)
            sc = rs/rm
            pn = [(c[0]+(p[0]-c[0])*sc, c[1]+(p[1]-c[1])*sc) for p in pts[:8]]
            nn.append(sum(min(d(gp[j], pn[k]) for k in range(8)) for j in range(8))/8.0)
            ordv.append(sum(d(gp[j], pn[j]) for j in range(8))/8.0)
    print("帧 %d  亮靶标配对 %d  漏检 %d" % (frames, n, miss))
    print("[靶标中心] 模型8点均值 vs 真值   中位=%6.1f p90=%6.1f px" % (med(ctr), pc(ctr,.9)))
    print("[符文中心] kpts[8](R心) vs 真值符心 中位=%6.1f p90=%6.1f px" % (med(rune), pc(rune,.9)))
    print("[缩放比  ] 模型半径/真值半径    中位=%6.3f p10=%.3f p90=%.3f" % (med(ratio), pc(ratio,.1), pc(ratio,.9)))
    print("[8点 等比缩放后] 最近邻 中位=%6.1f p90=%6.1f | 同序号 中位=%6.1f" % (med(nn), pc(nn,.9), med(ordv)))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hw26_rune_kpts_log.jsonl")
