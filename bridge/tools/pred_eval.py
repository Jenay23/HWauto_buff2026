import csv, math, statistics
PI=math.pi; DEG=180/PI
def wrap(a): return (a+PI)%(2*PI)-PI
rows=[]
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try: rows.append(dict(t=int(r["sim_t_ns"])/1e9, th=float(r["local_theta_rad"]),
                              c=int(r["control"]), fe=float(r["fire_err_rad"]),
                              lm=float(r["latency_ms"]), roll=float(r["roll_rad"])))
        except: pass
rows.sort(key=lambda r:r["t"])
# 只看有相位的时间段
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')

# 1) 瞄准误差 err=（开火帧）
fe=[r["fe"] for r in rows if r["fe"]>=0]
if fe: print("fire_err: p50=%.4f p90=%.4f p99=%.4f rad  (n=%d)"%(med(fe),pc(fe,.9),pc(fe,.99),len(fe)))
fe2=[x*DEG for x in fe]
print("         = p50=%.2f p90=%.2f p99=%.2f 度"% (med(fe2),pc(fe2,.9),pc(fe2,.99)))

# 2) 相位外推误差: 用最近 Tfit 秒拟合 [th0,a,w,phi]，预测 +H 秒，与实测比
import numpy as np
ts=np.array([r["t"] for r in rows]); th=np.unwrap(np.array([r["th"] for r in rows]))
KT=2.090
def fit_pred(i0,i1,H):
    t=ts[i0:i1]-ts[i0]; y=th[i0:i1]
    if len(t)<8 or t[-1]<0.3: return None
    # 粗网格
    best=None
    for a in np.linspace(0.780,1.045,6):
      for w in np.linspace(1.884,2.000,6):
        for phi in np.linspace(0,2*math.pi,8,endpoint=False):
          for sg in (-1,1):
            base=(KT-a)*t+(a/w)*(np.cos(phi)-np.cos(w*t+phi))
            # 解 th0
            th0=float(np.mean(y-sg*base))
            r=float(np.sqrt(np.mean((y-(th0+sg*base))**2)))
            if best is None or r<best[0]: best=(r,th0,a,w,phi,sg)
    r,th0,a,w,phi,sg=best
    tq=ts[i1-1]-ts[i0]+H
    pred=th0+sg*((KT-a)*tq+(a/w)*(np.cos(phi)-np.cos(w*tq+phi)))
    # 找 t+ H 处的实测
    tgt=ts[i1-1]+H
    j=int(np.searchsorted(ts,tgt))
    if j>=len(ts) or abs(ts[j]-tgt)>0.06: return None
    return abs(wrap(ts_dummy:=th[j]-pred))*DEG if False else abs(wrap(th[j]-pred))*DEG, r*DEG
for H in (0.10,0.15,0.30):
    errs=[]; inr=[]
    step=max(3,int(0.1/0.03))
    i=8
    while i<len(rows)-20:
        Tfit=0.8
        i0=i
        while i0>0 and ts[i]-ts[i0]<Tfit: i0-=1
        out=fit_pred(i0,i+1,H)
        if out: errs.append(out[0]); inr.append(out[1])
        i+=step
    if errs:
        print("H=%.2fs: 外推误差 中位=%.1f p90=%.1f 度 | 窗内 rms 中位=%.1f 度 (n=%d)"%(
            H,med(errs),pc(errs,.9),med(inr),len(errs)))
