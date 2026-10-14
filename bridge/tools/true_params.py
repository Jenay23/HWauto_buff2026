import json, math, numpy as np
FX=FY=1738.5; CX=720.0; CY=540.0
KMEAN=2.090
# 从真值日志算"真实相位轨迹"：用两片亮靶标的中点（对身份免疫、旋转速率与叶片相同）
rec=[]
for line in open("logs/20261014_223458_badwin.kpts.jsonl"):
    i=line.find(',')
    if i<0: continue
    try: r=json.loads(line[i+1:])
    except: continue
    sim=r.get("sim")
    if not sim: continue
    allc=np.array([t["center_cam"] for t in sim["targets"]])          # 5 靶标中心(相机系)
    lits=[np.array(t["center_cam"]) for t in sim["targets"] if t["state"]==1]
    if len(lits)<2: continue
    C=allc.mean(0)                                                    # 符心
    M=np.mean(lits,axis=0)                                            # 两片中点
    rec.append((int(line[:i])/1e9, C, M, allc))
print("有双亮片的帧:",len(rec))
# 用所有帧的 5 个中心拟合转盘平面（法向）
A=np.vstack([r[3]-r[1] for r in rec[:400]])
Vt=np.linalg.svd(A)[2]; n=Vt[2]; e1=Vt[0]; e2=Vt[1]
# 真实相位轨迹: 中点绕符心的面内角
t=np.array([r[0] for r in rec]); t-=t[0]
ang=np.array([math.atan2((r[2]-r[1])@e2, (r[2]-r[1])@e1) for r in rec])
# 半径(校验用)
rad=np.array([np.linalg.norm(r[2]-r[1]) for r in rec])
print("中点绕符心半径: 中位=%.3f m (理论=0.7*cos(dlt/2))"%np.median(rad))
# 解缠
un=np.unwrap(ang)
# 物理模型拟合（滑窗, 2s）
def fitwin(ts,ys):
    if len(ts)<20 or ts[-1]-ts[0]<1.0: return None
    best=None
    for a in np.linspace(0.780,1.045,6):
      for w in np.linspace(1.884,2.000,6):
        for phi in np.linspace(0,2*math.pi,8,endpoint=False):
          for sg in (-1,1):
            base=(KMEAN-a)*ts+(a/w)*(np.cos(phi)-np.cos(w*ts+phi))
            th0=float(np.mean(ys-sg*base))
            se=float(np.mean((ys-(th0+sg*base))**2))
            if best is None or se<best[0]: best=(se,th0,a,w,phi,sg)
    se,th0,a,w,phi,sg=best
    if se>0.30: return None        # rms>0.55rad 的窗判为不可信
    return a,w,sg,math.sqrt(se)*57.2958
outs=[]
for g in range(0,len(t)-60, 10):
    sel=(t>=t[g])&(t<t[g]+2.0)
    if sel.sum()<20: continue
    r=fitwin(t[sel]-t[g], un[sel])
    if r: outs.append((t[g],r[0],r[1],r[2],r[3]))
print("可拟合窗口数:",len(outs))
if outs:
    A_=np.array([o[1] for o in outs]); W_=np.array([o[2] for o in outs]); R_=np.array([o[4] for o in outs])
    print("【真实参数】a: 中位=%.3f  p10=%.3f p90=%.3f"%tuple(np.percentile(A_,[50,10,90])))
    print("【真实参数】w: 中位=%.3f  p10=%.3f p90=%.3f"%tuple(np.percentile(W_,[50,10,90])))
    print("真实拟合窗内 rms: 中位=%.2f deg"%np.median(R_))
    np.save("/tmp/true_params.npy", np.array(outs))
