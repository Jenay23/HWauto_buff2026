import csv, json, math, bisect
FX=FY=1738.5; CX=720.0; CY=540.0
aim=[]
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try:
            if int(r["control"])!=1: continue
            ax,ay=float(r["aim_x"]),float(r["aim_y"])
            if ax<0: continue
            aim.append((int(r["sim_t_ns"]), ax, ay))
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
def nearest(ts,tol=6e7):
    j=bisect.bisect_left(tss,ts); best=None
    for k in (j-1,j,j+1):
        if 0<=k<len(tss) and abs(tss[k]-ts)<=tol:
            if best is None or abs(tss[k]-ts)<abs(best-ts): best=tss[k]
    return best
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
print("%8s %10s %10s"%("lead(s)","|Δ|中位","|Δ|p90"))
for lead in [0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.536,0.60,0.70,0.80]:
    d=[]
    for ts,ax,ay in aim:
        k=nearest(ts+int(lead*1e9))
        if k: d.append(min(math.hypot(ax-x,ay-y) for x,y in lit[k]))
    if d: print("%8.3f %10.1f %10.1f"%(lead,med(d),sorted(d)[int(.9*len(d))]))
