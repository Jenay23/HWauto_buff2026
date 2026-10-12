#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线评估：用真实 [bridge] 日志窗口比较"物理约束模型"与现有方案的多步预测精度。
只做预测精度评测，不接模拟器。"""
import math, re, sys, glob
from pathlib import Path

PI = math.pi; DEG = 180.0/PI
ALO, AHI = 0.780, 1.045
WLO, WHI = 1.884, 2.000
MEAN = 2.090

def wrap(a):
    return (a + PI) % (2*PI) - PI

RE = re.compile(r"\[bridge\]\s+#(?P<k>\d+)\s+age=(?P<age>[-\d.]+)ms\s+det=(?P<det>\d+)\s+sel=(?P<sel>\d+)\s+solved=(?P<sol>\d+).*?st=(?P<st>\d+)\s+dis=(?P<dis>[-\d.]+)\s+roll=(?P<roll>[-\d.]+).*?lat=(?P<lat>[-\d.]+)ms")

def parse(p):
    out=[]
    for line in Path(p).read_text(errors="ignore").splitlines():
        m=RE.search(line)
        if not m: continue
        out.append((int(m.group("k")), float(m.group("roll")), int(m.group("det")), int(m.group("sol")), float(m.group("lat"))))
    return out

def windows(fr, max_gap=3):
    sel=[f for f in fr if f[2]>0 and f[3]==1]
    wins=[]; cur=[]; prev=None
    for f in sel:
        if prev is not None and f[0]-prev>max_gap:
            if len(cur)>=6: wins.append(cur)
            cur=[]
        cur.append(f); prev=f[0]
    if len(cur)>=6: wins.append(cur)
    return wins

def unwrap(ys):
    out=[]; acc=0.0; prev=None
    for y in ys:
        if prev is not None: acc+=wrap(y-prev)
        prev=y; out.append(acc)
    return out

# ---------- 物理约束模型: theta = th0 + s*[(2.09-a)t + (a/w)(cos(phi)-cos(w t+phi))] ----------
def phys_eval(par, ts, s):
    th0,a,w,phi=par
    return [th0 + s*((MEAN-a)*t + (a/w)*(math.cos(phi)-math.cos(w*t+phi))) for t in ts]

def phys_fit(ts, ys, coarse_s):
    s = coarse_s
    best=None
    # 粗网格初始化 (a,w,phi) + 线性 th0
    for ia in range(5):
        a=ALO+(AHI-ALO)*ia/4
        for iw in range(6):
            w=WLO+(WHI-WLO)*iw/5
            for ip in range(12):
                phi=2*PI*ip/12
                pred=[s*((MEAN-a)*t+(a/w)*(math.cos(phi)-math.cos(w*t+phi))) for t in ts]
                mu=sum(ys)/len(ys); mp=sum(pred)/len(pred)
                th0=mu-mp
                se=sum((y-(p+th0))**2 for y,p in zip(ys,pred))
                if best is None or se<best[0]: best=(se,[th0,a,w,phi])
    par=list(best[1])
    # LM
    lam=1e-3
    for it in range(60):
        r=[y-p for y,p in zip(ys, phys_eval(par,ts,s))]
        # robust weights
        ar=sorted(abs(x) for x in r); sig=max(1e-4, 1.4826*ar[len(ar)//2])
        wt=[1.0/math.sqrt(1.0+(x/(2.5*sig))**2) for x in r]
        J=[]
        for t in ts:
            th0,a,w,phi=par
            base=(MEAN-a)*t+(a/w)*(math.cos(phi)-math.cos(w*t+phi))
            dth0=1.0
            da=s*(-t+(1.0/w)*(math.cos(phi)-math.cos(w*t+phi)))
            dw=s*a*( t*math.sin(w*t+phi)/w - (math.cos(phi)-math.cos(w*t+phi))/(w*w) )
            dphi=s*(a/w)*(-math.sin(phi)+math.sin(w*t+phi))
            J.append([dth0,da,dw,dphi])
        A=[[0.0]*4 for _ in range(4)]; b=[0.0]*4
        for i in range(len(ts)):
            for p in range(4):
                wi=wt[i]
                b[p]+=wi*wi*J[i][p]*r[i]
                for q in range(4): A[p][q]+=wi*wi*J[i][p]*J[i][q]
        for p in range(4): A[p][p]*=(1.0+lam); A[p][p]+=1e-12
        d=solve(A,b)
        if d is None: break
        new=[par[i]+d[i] for i in range(4)]
        new[1]=min(AHI,max(ALO,new[1])); new[2]=min(WHI,max(WLO,new[2]))
        se0=sum(x*x for x in r)
        se1=sum((y-p)**2 for y,p in zip(ys,phys_eval(new,ts,s)))
        if se1<se0:
            par=new; lam=max(lam*0.4,1e-6)
        else:
            lam=min(lam*4,1e3)
        if max(abs(x) for x in d)<1e-9: break
    return par

def solve(A,b):
    n=len(b); M=[row[:] for row in A]; v=b[:]
    for i in range(n):
        p=max(range(i,n), key=lambda r: abs(M[r][i]))
        M[i],M[p]=M[p],M[i]; v[i],v[p]=v[p],v[i]
        if abs(M[i][i])<1e-15: return None
        for r in range(i+1,n):
            f=M[r][i]/M[i][i]
            for c in range(i,n): M[r][c]-=f*M[i][c]
            v[r]-=f*v[i]
    x=[0.0]*n
    for i in range(n-1,-1,-1):
        x[i]=(v[i]-sum(M[i][j]*x[j] for j in range(i+1,n)))/M[i][i]
    return x

def lsq(basis, ys):
    m=len(basis[0]); A=[[0.0]*m for _ in range(m)]; v=[0.0]*m
    for b,y in zip(basis,ys):
        for i in range(m):
            v[i]+=b[i]*y
            for j in range(m): A[i][j]+=b[i]*b[j]
    for i in range(m): A[i][i]+=1e-12
    return solve(A,v)

# csine baseline (mode0 style), y 已经是解缠后的
def csine_fit(ts, ys):
    t0=ts[-1]; xs=[t-t0 for t in ts]
    coarse=(ys[-1]-ys[0])/(xs[-1]-xs[0]) if abs(xs[-1]-xs[0])>1e-9 else 0.0
    sgn=-1.0 if coarse<0 else 1.0
    best=None
    for ia in range(17):
        a=ALO+(AHI-ALO)*ia/16
        b=MEAN-a
        for iw in range(9):
            w=WLO+(WHI-WLO)*iw/8
            yv=[y-sgn*b*x for x,y in zip(xs,ys)]
            basis=[[1.0, sgn*(1-math.cos(w*x)), sgn*math.sin(w*x)] for x in xs]
            c=lsq(basis,yv)
            if c is None: continue
            se=sum((y-(sgn*b*x+c[0]+sgn*c[1]*(1-math.cos(w*x))+sgn*c[2]*math.sin(w*x)))**2 for x,y in zip(xs,ys))
            if best is None or se<best[0]: best=(se,a,b,w,sgn,c)
    if best is None: return None
    _,a,b,w,sgn,c=best
    def pred(tf):
        x=tf-t0
        return sgn*b*x+c[0]+sgn*c[1]*(1-math.cos(w*x))+sgn*c[2]*math.sin(w*x)
    return pred

def free_fit(ts, ys):
    t0=ts[-1]; xs=[t-t0 for t in ts]
    best=None
    for gi in range(33):
        w=WLO+(WHI-WLO)*gi/32
        basis=[[1.0,x,math.sin(w*x),math.cos(w*x)] for x in xs]
        c=lsq(basis,ys)
        if c is None: continue
        se=sum((y-(c[0]+c[1]*x+c[2]*math.sin(w*x)+c[3]*math.cos(w*x)))**2 for x,y in zip(xs,ys))
        if best is None or se<best[0]: best=(se,c,w,t0)
    if best is None: return None
    _,c,w,t0=best
    return lambda tf:(c[0]+c[1]*(tf-t0)+c[2]*math.sin(w*(tf-t0))+c[3]*math.cos(w*(tf-t0)))

def lin_fit(ts,ys):
    t0=ts[-1]; xs=[t-t0 for t in ts]; n=len(xs)
    sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    den=n*sxx-sx*sx
    if abs(den)<1e-12: return lambda tf: ys[-1]
    a1=(n*sxy-sx*sy)/den; a0=(sy-a1*sx)/n
    return lambda tf: a0+a1*(tf-t0)

def main():
    paths=sys.argv[1:] or sorted(glob.glob("logs/*.log"), key=lambda p:Path(p).stat().st_mtime, reverse=True)[:12]
    allw=[]; lats=[]
    for p in paths:
        fr=parse(p)
        if len(fr)<20: continue
        lats += [f[4] for f in fr]
        ws=windows(fr)
        allw.append((p,ws))
    lats.sort(); DT=(lats[len(lats)//2]/1000.0) if lats else 1/30.0
    print(f"使用 dt={DT*1000:.1f}ms（lat 中位）；窗口数={sum(len(ws) for _,ws in allw)}")
    # dt: 用窗口内相邻帧号差 = 1，时间用帧号 * 假设 dt；用另一估计：整段帧数与时长未知，这里直接用 1/30
    HOR=[6,9,12]   # 0.2/0.3/0.4 s
    TFIT=[0.8,1.2]
    methods=["lin","csine","free","phys"]
    for T in TFIT:
        for H in HOR:
            err={m:[] for m in methods}
            nwin=0
            for p,ws in allw:
                for w in ws:
                    ys=unwrap([f[1] for f in w])
                    ts=[i*DT for i in range(len(w))]
                    # fit 到 T 秒，预测到 T+H*DT
                    import bisect
                    tv=[(f[0]-w[0][0])*DT for f in w]
                    s=bisect.bisect_right(tv, T)-1
                    if s<5: continue
                    if s+H>=len(w): continue
                    fts, fys=tv[:s+1], ys[:s+1]
                    tf=tv[s+H]
                    tgt=ys[s+H]
                    coarse=(fys[-1]-fys[0])/(fts[-1]-fts[0]+1e-9)
                    sgn=-1.0 if coarse<0 else 1.0
                    try:
                        ph=phys_fit(fts,fys,sgn)
                        pv=phys_eval(ph,[tf],sgn)[0]
                        err["phys"].append(abs(wrap(pv-tgt))*DEG)
                    except Exception as e:
                        pass
                    fn=csine_fit(fts,fys)
                    if fn: err["csine"].append(abs(wrap(fn(tf)-tgt))*DEG)
                    fn=free_fit(fts,fys)
                    if fn: err["free"].append(abs(wrap(fn(tf)-tgt))*DEG)
                    fn=lin_fit(fts,fys)
                    err["lin"].append(abs(wrap(fn(tf)-tgt))*DEG)
                    nwin+=1
            print(f"Tfit={T:.2f}s ({T*1000:.0f}ms)  horizon={H} ({H*DT*1000:.0f}ms)  windows={nwin}")
            for m in methods:
                v=sorted(err[m])
                if len(v)<5: print(f"   {m:>6}: n={len(v)}"); continue
                med=v[len(v)//2]; p90=v[min(len(v)-1,int(0.9*len(v)))]
                print(f"   {m:>6}: 中位 {med:6.2f}°  p90 {p90:6.2f}°  (n={len(v)})")
    return

if __name__=="__main__":
    main()
