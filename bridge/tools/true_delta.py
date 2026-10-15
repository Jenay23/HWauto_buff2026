import json, math, numpy as np, collections
for f in ["logs/20261015_195559_fire12.kpts.jsonl","logs/20261015_200727_fire12.kpts.jsonl"]:
    recs=[]
    for line in open(f):
        i=line.find(',')
        if i<0: continue
        try: r=json.loads(line[i+1:])
        except: continue
        s=r.get("sim")
        if not s: continue
        ids=[(t["target_id"], np.array(t["center_cam"])) for t in s["targets"] if t["state"]==1]
        allc=np.array([t["center_cam"] for t in s["targets"]])
        if len(ids)!=2: continue
        C=allc.mean(0)
        v1=ids[0][1]-C; v2=ids[1][1]-C
        # 两片夹角（相机系里直接算，够用）
        c=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        ang=math.degrees(math.acos(max(-1,min(1,c))))
        recs.append((ids[0][0],ids[1][0],ang))
    if not recs: print(f,"无数据"); continue
    print("=== %s  (%d 帧) ==="%(f.split("/")[-1],len(recs)))
    print("  亮片 target_id 组合:", dict(collections.Counter((a,b) for a,b,_ in recs)))
    angs=[x[2] for x in recs]
    print("  两片夹角(度): p10=%.1f p50=%.1f p90=%.1f"%(np.percentile(angs,10),np.percentile(angs,50),np.percentile(angs,90)))
    # 夹角随时间变化
    d=[abs(a2-a1) for a1,a2 in zip(angs,angs[1:])]
    print("  夹角逐帧变化: 中位=%.2f p90=%.2f max=%.2f"%(np.percentile(d,50),np.percentile(d,90),max(d)))
    # 每 100 帧采样
    print("  夹角轨迹(每 200 帧):", [round(angs[i],1) for i in range(0,len(angs),200)])
