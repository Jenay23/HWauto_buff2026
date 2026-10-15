import csv, json, math, bisect
FX=FY=1738.5; CX=720.0; CY=540.0
aim=[]
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try:
            if int(r["control"])!=1: continue
            ax,ay=float(r["aim_x"]),float(r["aim_y"])
            if ax<0: continue
            aim.append((int(r["sim_t_ns"]), ax, ay, float(r["lead_s"]), float(r["fly_s"])))
        except: pass
lit={}
for line in open("logs/20261015_141623_fire12.kpts.jsonl"):
    i=line.find(',')
    if i<0: continue
    try: rec=json.loads(line[i+1:])
    except: continue
    s=rec.get("sim")
    if not s: continue
    p=[]
    for t in s["targets"]:
        if t["state"]==1:
            c=t["center_cam"]; z=c[2]
            if z>1e-6: p.append((FX*c[0]/z+CX, FY*c[1]/z+CY, z))
    if p: lit[int(line[:i])]=p
tss=sorted(lit)
def nearest(ts,tol=6e7):
    j=bisect.bisect_left(tss,ts); best=None
    for k in (j-1,j,j+1):
        if 0<=k<len(tss) and abs(tss[k]-ts)<=tol:
            if best is None or abs(tss[k]-ts)<abs(best-ts): best=tss[k]
    return best
du=[];dv=[];drop=[]
for ts,ax,ay,lead,fly in aim:
    k=nearest(ts+int(lead*1e9))
    if not k: continue
    # 取"水平最近"的那个靶标（下坠只影响 v）
    best=min(lit[k],key=lambda q:abs(ax-q[0]))
    du.append(ax-best[0]); dv.append(ay-best[1])
    # 理论下坠(px): drop = 0.5*g*t^2, 角度=drop/z, px=角度*FX
    z=best[2]; d=0.5*9.81*fly*fly
    drop.append(d/z*FX)
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')
print("样本 %d"%len(du))
print("Δu (水平): 中位=%+.1f p10=%+.1f p90=%+.1f | |Δu| 中位=%.1f p90=%.1f px"%(
    med(du),pc(du,.1),pc(du,.9),med([abs(x) for x in du]),pc([abs(x) for x in du],.9)))
print("Δv (垂直): 中位=%+.1f p10=%+.1f p90=%+.1f px"%(med(dv),pc(dv,.1),pc(dv,.9)))
print("理论下坠补偿(px): 中位=%.1f  --> 期望 Δv ≈ -该值(往上瞄)"%(med(drop)))
print()
print("=> 若 |Δu| 中位 很小 且 Δv ≈ -下坠，则瞄准是精准的，之前的 100px 全是下坠补偿")
