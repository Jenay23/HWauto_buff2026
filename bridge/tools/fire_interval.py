import csv, statistics
rows=[(float(r["wall_s"]), int(r["launch"]), int(r["hits"])) for r in csv.DictReader(open("logs/20261015_135649_fire12.rtstat.csv"))]
# 找 launch 阶跃时刻（可能一次跳多）
shots=[]; hits=[]
for i in range(1,len(rows)):
    d=rows[i][1]-rows[i-1][1]
    for k in range(d): shots.append(rows[i][0])
    dh=rows[i][2]-rows[i-1][2]
    for k in range(dh): hits.append(rows[i][0])
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')
iv=sorted(b-a for a,b in zip(shots,shots[1:]))
print("发弹 %d 次"%len(shots))
print("发弹间隔 s: p10=%.2f 中位=%.2f p90=%.2f"%(pc(iv,.1),med(iv),pc(iv,.9)))
import collections
c=collections.Counter(round(x,1) for x in iv)
print("间隔分布(四舍五入到 0.1s，取前 8):", c.most_common(8))
print()
print("=> 模拟器冷却=0.833s；若间隔大量落在 0.8~0.9，说明模拟器确实是限制项")
print("=> 若大量 >1.0s，说明瓶颈在【桥的开火门】(lock/对准)")
# 每分钟命中
print()
print("每分钟命中:")
t0=rows[0][0]
buckets={}
for t,l,h in rows:
    b=int((t-t0)//60)
    buckets[b]=h
prev=0
for b in sorted(buckets):
    if b>10: break
    print("  第 %2d 分钟: 累计 hits=%d (+%d)"%(b+1,buckets[b],buckets[b]-prev)); prev=buckets[b]
