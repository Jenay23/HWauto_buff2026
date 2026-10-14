import csv, json, math, statistics
FX=FY=1738.5; CX=720.0; CY=540.0
# 1) 读 CSV 的打击点
aim={}
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        if int(r["control"])==1:
            aim[int(r["sim_t_ns"])]=(float(r["aim_x"]), float(r["aim_y"]))
# 2) 读真值 kpts 日志
lit={}
for line in open("logs/20261014_223458_badwin.kpts.jsonl"):
    i=line.find(',')
    if i<0: continue
    try: rec=json.loads(line[i+1:])
    except: continue
    sim=rec.get("sim")
    if not sim: continue
    pts=[]
    for t in sim["targets"]:
        if t["state"]!=1: continue
        c=t["center_cam"]; z=c[2]
        if z>1e-6: pts.append((FX*c[0]/z+CX, FY*c[1]/z+CY))
    if pts: lit[int(line[:i])]=pts
# 3) 对每个 control 帧，打击点到最近真值靶标中心的距离
d=[]; dd=[]
for ts,(ax,ay) in aim.items():
    # 找最近的 kpts 时间戳
    best=None
    for k in (ts-40000000, ts-20000000, ts, ts+20000000, ts+40000000):
        for q in (k, k-6000000, k+6000000):
            if q in lit: best=q; break
        if best: break
    if not best: continue
    ds=[math.hypot(ax-p[0], ay-p[1]) for p in lit[best]]
    if ds: d.append(min(ds))
import numpy as np
d.sort(); n=len(d)
print("配对 %d 帧"%n)
print("打击点 -> 最近真值靶标中心 距离 px: p10=%.0f p25=%.0f p50=%.0f p75=%.0f p90=%.0f p99=%.0f"%(
    d[int(.1*n)],d[int(.25*n)],d[n//2],d[int(.75*n)],d[int(.9*n)],d[min(n-1,int(.99*n))]))
# 参考：真实的"提前量"造成的固有偏移 = lead*omega*R
print("参考: 提前量 lead~0.1s, om~2.0, R=0.7 -> 固有偏移 ≈ 0.14 m -> 7m 处 ≈ %.0f px"%(0.14*1738.5/7.0))
print("      => 打击点本来就该【超前】靶标约这个距离，所以中位数接近它是正常的")
