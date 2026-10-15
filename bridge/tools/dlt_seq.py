import re
pat = re.compile(r"dlt=([-\d.]+) cand=.*?local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg.*?om=([-\d.]+) A=([-\d.]+)")
f="logs/20261015_200727_fire12.log"
rows=[(float(m.group(1)),int(m.group(2)),int(m.group(3)),float(m.group(4)),float(m.group(5)),float(m.group(6)))
      for m in pat.finditer(open(f,errors="ignore").read())]
print("共 %d 行；dlt / (n,rdy,rms,om,A):"%len(rows))
seq=[]
for i,(dlt,n,rdy,rms,om,A) in enumerate(rows):
    seq.append(round(dlt))
print("dlt 序列:")
for i in range(0,len(seq),1):
    if i%20==0: print("  [%3d]"%i, end="")
    print(" %3d"%seq[i], end="")
    if i%20==19: print()
if len(seq)%20: print()
# 统计连续相同段
runs=[];cur=1
for i in range(1,len(seq)):
    if seq[i]==seq[i-1]: cur+=1
    else: runs.append(cur); cur=1
runs.append(cur)
print("\n相同 dlt 的连续段长: 中位=%.0f 均值=%.1f 最長=%d 段数=%d"%(sorted(runs)[len(runs)//2],sum(runs)/len(runs),max(runs),len(runs)))
