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
            if z>1e-6: p.append((FX*c[0]/z+CX, FY*c[1]/z+CY))
    if p: lit[int(line[:i])]=p
tss=sorted(lit)
def nearest(ts, tol=6e7):
    j=bisect.bisect_left(tss,ts); best=None
    for k in (j-1,j,j+1):
        if 0<=k<len(tss) and abs(tss[k]-ts)<=tol:
            if best is None or abs(tss[k]-ts)<abs(best-ts): best=tss[k]
    return best
d_lead=[]
for ts,ax,ay,lead,fly in aim:
    k=nearest(ts+int(lead*1e9))
    if k: d_lead.append(min(math.hypot(ax-x,ay-y) for x,y in lit[k]))
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')
n=len(d_lead)
print("样本 %d"%n)
print("★ 打击点 -> 靶标在【lead=提前量时刻】的真实位置 (px):")
print("   p10=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f p99=%.1f"%(pc(d_lead,.1),pc(d_lead,.25),med(d_lead),pc(d_lead,.75),pc(d_lead,.9),pc(d_lead,.99)))
R=0.15*FX/7.0
print("   靶标像素半径≈%.0f px -> 落在靶面内(p50<R) 占比=%.1f%%"%(R,100*sum(1 for x in d_lead if x<R)/n))
print("   <=R 占比 %.1f%% | <=2R %.1f%% | <=74(整个板) %.1f%%"%(100*sum(1 for x in d_lead if x<=R)/n,100*sum(1 for x in d_lead if x<=2*R)/n,100*sum(1 for x in d_lead if x<=74)/n))
