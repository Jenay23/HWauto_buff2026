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
    lit[int(line[:i])]=p
tss=sorted(lit)
def nearest(ts,tol=6e7):
    j=bisect.bisect_left(tss,ts); best=None
    for k in (j-1,j,j+1):
        if 0<=k<len(tss) and abs(tss[k]-ts)<=tol:
            if best is None or abs(tss[k]-ts)<abs(best-ts): best=tss[k]
    return best
d=[]; du=[]; dv=[]
for ts,ax,ay,lead,fly in aim:
    k=nearest(ts+int(lead*1e9))
    if not k: continue
    # 把"下坠补偿"加回去：aim_v + drop_px 应该等于靶标 v
    if not lit[k]: continue
    best=None
    for (x,y,z) in lit[k]:
        drop=0.5*9.81*fly*fly/z*FX
        e=math.hypot(ax-x, (ay+drop)-y)
        if best is None or e<best[0]: best=(e,ax-x,(ay+drop)-y,drop)
    d.append(best[0]); du.append(best[1]); dv.append(best[2])
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')
n=len(d)
print("样本 %d （已加回弹道下坠补偿）"%n)
print("★ 打击点 -> 靶标在 t+lead 的真实位置:")
print("   总误差 px: p10=%.0f p25=%.0f p50=%.0f p75=%.0f p90=%.0f"%(pc(d,.1),pc(d,.25),med(d),pc(d,.75),pc(d,.9)))
print("   Δu 水平: 中位=%+.1f |Δu|中位=%.1f (p90=%.1f)"%(med(du),med([abs(x) for x in du]),pc([abs(x) for x in du],.9)))
print("   Δv 垂直: 中位=%+.1f |Δv|中位=%.1f (p90=%.1f)"%(med(dv),med([abs(x) for x in dv]),pc([abs(x) for x in dv],.9)))
R=0.15*FX/7.0
print("   靶标像素半径≈%.0f px -> 落在靶面内(<=R) 占 %.1f%%, <=1.5R 占 %.1f%%, <=2R 占 %.1f%%"%(
    R,100*sum(1 for x in d if x<=R)/n,100*sum(1 for x in d if x<=1.5*R)/n,100*sum(1 for x in d if x<=2*R)/n))
