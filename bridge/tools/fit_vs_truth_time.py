import csv, json, math, statistics
cs={}
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try: cs[int(r["sim_t_ns"])]=(float(r["local_amp"]), float(r["local_omega"]))
        except: pass
tr=[]
for line in open("logs/20261014_224624_fitcheck.kpts.jsonl"):
    i=line.find(',')
    if i<0: continue
    try: rec=json.loads(line[i+1:])
    except: continue
    s=rec.get("sim")
    if not s or s.get("true_a") is None: continue
    tr.append((int(line[:i]), float(s["true_a"]), float(s["true_omega"])))
tr.sort()
# 按 true_a 变化切段（每段=一次激活，真值恒定）
segs=[]; cur=[]
for x in tr:
    if cur and abs(x[1]-cur[-1][1])>1e-6:
        segs.append(cur); cur=[]
    cur.append(x)
if cur: segs.append(cur)
segs=[s for s in segs if len(s)>60]
print("激活段数=%d"%len(segs))
# 每段内: 误差 vs 已激活时长
buckets={0.5:[],1.0:[],1.5:[],2.5:[],99:[]}
for s in segs:
    t0=s[0][0]
    for ts,ta,tw in s:
        if ts not in cs: continue
        a,w=cs[ts]
        if a==0.0: continue
        el=(ts-t0)/1e9
        for b in buckets:
            if el<=b: buckets[b].append(abs(a-ta)); break
def med(x): x=sorted(x); return x[len(x)//2] if x else float('nan')
print("\n按【已激活时长】看 |Δa|:")
for b in sorted(buckets):
    v=buckets[b]
    if v: print("  <= %.1fs : |Δa| 中位=%.4f  n=%d"%(b,med(v),len(v)))
# 每段内 a 误差的收敛: 段内前 20% vs 后 20%
print("\n每段内 |Δa| 的变化:")
first=[];last=[]
for s in segs:
    vals=[]
    t0=s[0][0]
    for ts,ta,tw in s:
        if ts in cs and cs[ts][0]!=0.0: vals.append((ts,abs(cs[ts][0]-ta)))
    if len(vals)>20:
        k=max(1,len(vals)//5)
        first+= [v for _,v in vals[:k]]; last+=[v for _,v in vals[-k:]]
print("  段前20%%: |Δa| 中位=%.4f   段后20%%: |Δa| 中位=%.4f"%(med(first),med(last)))
