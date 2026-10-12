#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析真实 CSV：量化【实际喂进局部 tracker 的相位】的质量，并做 held-out 物理拟合预测。
新 CSV 末尾两列: local_theta_rad, local_chord (1=弦相位)。"""
import csv, math, importlib.util, bisect, sys
PI=math.pi; DEG=180.0/PI
def wrap(a): return (a+PI)%(2*PI)-PI
spec=importlib.util.spec_from_file_location("fe","tools/fit_eval.py")
fe=importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)

def main(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append(dict(t=int(r["sim_t_ns"])/1e9, roll=float(r["roll_rad"]),
                                 theta=float(r.get("local_theta_rad",r["roll_rad"])),
                                 ch=int(float(r.get("local_chord",0) or 0)),
                                 det=int(r["n_det"]), sol=int(r["solved"]),
                                 ctrl=int(r["control"]), fire=int(r["fire"])))
            except Exception: pass
    rows=[r for r in rows if r["t"]>0]; rows.sort(key=lambda r:r["t"])
    if not rows: print("no rows"); return
    print(f"CSV rows={len(rows)} span={rows[-1]['t']-rows[0]['t']:.1f}s ctrl={sum(r['ctrl'] for r in rows)} fire={sum(r['fire'] for r in rows)}")
    lit=[r for r in rows if r["det"]>0 and r["sol"]==1]
    print(f"lit={len(lit)}  其中 ch=1 比例={100.0*sum(r['ch'] for r in lit)/max(1,len(lit)):.1f}%")
    wins=[];cur=[];prev=None
    for r in lit:
        if prev is not None and r["t"]-prev>0.25:
            if len(cur)>=10: wins.append(cur)
            cur=[]
        cur.append(r);prev=r["t"]
    if len(cur)>=10: wins.append(cur)
    for key,lab in (("roll","原始 PnP roll"),("theta","实际喂入 theta")):
        flips=0;tot=0
        for w in wins:
            for i in range(1,len(w)):
                dt=w[i]["t"]-w[i-1]["t"]
                if dt<=0: continue
                tot+=1
                if abs(wrap(w[i][key]-w[i-1][key])/dt)>2.090*1.2: flips+=1
        print(f"[{lab}] 超物理上限步占比 {100.0*flips/max(1,tot):.1f}%  ({flips}/{tot})")
    for key,lab in (("roll","roll"),("theta","theta")):
        for T in (0.8,1.2):
            errs=[]
            for w in wins:
                tv=[r["t"]-w[0]["t"] for r in w]; ys=[];acc=0.0;prev=None
                for r in w:
                    if prev is not None: acc+=wrap(r[key]-prev)
                    prev=r[key];ys.append(acc)
                s=bisect.bisect_right(tv,T)-1
                if s<6: continue
                hs=[w[i+1]["t"]-w[i]["t"] for i in range(len(w)-1)]
                dt=sorted(hs)[len(hs)//2] if hs else 1/30
                target=s+max(1,int(round(0.30/dt)))
                if target>=len(w): continue
                coarse=(ys[s]-ys[0])/(tv[s]-tv[0]+1e-9); sgn=-1.0 if coarse<0 else 1.0
                try:
                    par=fe.phys_fit(tv[:s+1],ys[:s+1],sgn); pv=fe.phys_eval(par,[tv[target]],sgn)[0]
                    errs.append(abs(wrap(pv-ys[target]))*DEG)
                except Exception: pass
            if errs:
                errs.sort(); print(f"  phys[{lab}] Tfit={T}s -> +0.3s: 中位 {errs[len(errs)//2]:.1f}° p90 {errs[min(len(errs)-1,int(0.9*len(errs)))]:.1f}° (n={len(errs)})")
main(sys.argv[1] if len(sys.argv)>1 else "hw26_bridge_log.csv")
