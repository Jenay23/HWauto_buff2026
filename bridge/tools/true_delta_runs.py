import json, math, numpy as np
f="logs/20261015_200727_fire12.kpts.jsonl"
angs=[]
for line in open(f):
    i=line.find(',')
    if i<0: continue
    try: r=json.loads(line[i+1:])
    except: continue
    s=r.get("sim")
    if not s: continue
    c=[np.array(t["center_cam"]) for t in s["targets"] if t["state"]==1]
    if len(c)!=2: continue
    v1=c[0]-c[1]
    # 用符面内角（简单用相机系的夹角）
    ang=math.degrees(math.acos(max(-1,min(1,float(np.dot(c[0]-np.mean([t["center_cam"] for t in s["targets"]],axis=0),
                                                             c[1]-np.mean([t["center_cam"] for t in s["targets"]],axis=0))
        /(np.linalg.norm(c[0]-np.mean([t["center_cam"] for t in s["targets"]],axis=0))*
          np.linalg.norm(c[1]-np.mean([t["center_cam"] for t in s["targets"]],axis=0)))))))) 
    angs.append(144 if ang>108 else 72)
runs=[];cur=1
for i in range(1,len(angs)):
    if angs[i]==angs[i-1]: cur+=1
    else: runs.append(cur); cur=1
runs.append(cur)
print("真值 Δ 的连续段长: 中位=%.0f 均值=%.1f 最長=%d 段数=%d (共 %d 帧)"%(
    sorted(runs)[len(runs)//2],sum(runs)/len(runs),max(runs),len(runs),len(angs)))
print("每段约 %.1f 秒 (33fps)"%(sum(runs)/len(runs)/33))
print("对桥的 [dual] 采样(每10帧): 期望每 %.1f 行才变一次"%((sum(runs)/len(runs))/10))
print()
print("对比: 桥实测 dlt 连续段长 中位=4 行 -> 每 %.0f 帧就变一次"%(4*10))
