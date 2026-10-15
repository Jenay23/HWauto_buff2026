import csv, json, math, statistics
# 1) CSV: 拟合出的 a/w + 开火
cs={}
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try: cs[int(r["sim_t_ns"])]=(float(r["local_amp"]), float(r["local_omega"]), int(r["fire"]))
        except: pass
# 2) 真值
tr=[]
for line in open("logs/20261015_135649_fire12.kpts.jsonl"):
    i=line.find(',')
    if i<0: continue
    try: rec=json.loads(line[i+1:])
    except: continue
    s=rec.get("sim")
    if not s or s.get("true_a") is None: continue
    tr.append((int(line[:i]), float(s["true_a"]), float(s["true_omega"])))
tr.sort()
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')

J=[]
for ts,ta,tw in tr:
    if ts in cs:
        a,w,fi=cs[ts]
        if a!=0.0: J.append((ts,ta,tw,a-ta,w-tw,fi))
print("可比帧 %d"%len(J))
da=[x[3] for x in J]; dw=[x[4] for x in J]
print("\n【a 误差】(拟合-真值): 中位=%+.4f 均值=%+.4f | |误差| 中位=%.4f p90=%.4f max=%.4f"%(
    med(da),sum(da)/len(da),med([abs(x) for x in da]),pc([abs(x) for x in da],.9),max(abs(x) for x in da)))
print("【w 误差】(拟合-真值): 中位=%+.4f 均值=%+.4f | |误差| 中位=%.4f p90=%.4f max=%.4f"%(
    med(dw),sum(dw)/len(dw),med([abs(x) for x in dw]),pc([abs(x) for x in dw],.9),max(abs(x) for x in dw)))

# 按激活段（true_a 变化切）看收敛
segs=[]; cur=[]
for x in tr:
    if cur and abs(x[1]-cur[-1][1])>1e-6: segs.append(cur); cur=[]
    cur.append(x)
if cur: segs.append(cur)
segs=[s for s in segs if len(s)>60]
print("\n激活段数(>60帧)=%d"%len(segs))
buckets={0.5:[],1.0:[],1.5:[],2.5:[],5.0:[],99:[]}
for s in segs:
    t0=s[0][0]
    for ts,ta,tw in s:
        if ts in cs and cs[ts][0]!=0.0:
            el=(ts-t0)/1e9
            for b in buckets:
                if el<=b: buckets[b].append(abs(cs[ts][0]-ta)); break
print("|Δa| vs 已激活时长:")
for b in sorted(buckets):
    v=buckets[b]
    if v: print("   <=%4.1fs: 中位=%.4f n=%d"%(b,med(v),len(v)))

# 3) 开火帧 vs 非开火帧的误差 —— 打空是否伴随大误差？
f_on=[abs(x[3]) for x in J if x[5]==1]; f_off=[abs(x[3]) for x in J if x[5]==0]
print("\n|Δa| 在【开火帧】=%.4f (n=%d) vs 【非开火帧】=%.4f (n=%d)"%(med(f_on),len(f_on),med(f_off),len(f_off)))
wo=[abs(x[4]) for x in J if x[5]==1]; wf=[abs(x[4]) for x in J if x[5]==0]
print("|Δw| 在【开火帧】=%.4f vs 【非开火帧】=%.4f"%(med(wo),med(wf)))

# 4) 参数误差 -> 相位预测误差（0.54s 提前量）
lead=0.536
pa=[abs(x[3])*lead*57.2958 for x in J]; pw=[abs(x[4])*lead*57.2958 for x in J]
print("\n换算到 0.54s 提前量上的相位误差（只算 a 的 -t 项）:")
print("  来自 Δa: 中位=%.2f° p90=%.2f°"%(med(pa),pc(pa,.9)))
print("  来自 Δw: 中位=%.2f° p90=%.2f°"%(med(pw),pc(pw,.9)))
