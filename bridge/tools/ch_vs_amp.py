import csv, statistics
rows=[]
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try:
            rows.append(dict(ch=int(r["local_chord"]), amp=float(r["local_amp"]),
                             om=float(r["local_omega"]), th=float(r["local_theta_rad"]),
                             c=int(r["control"])))
        except: pass
def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))] if a else float('nan')
for ch in (0,2):
    s=[r for r in rows if r["ch"]==ch and r["amp"]>0]
    if not s: continue
    amps=[r["amp"] for r in s]
    print("ch=%d: n=%5d (%.0f%%) | amp 中位=%.3f p90=%.3f | amp>=1.044 占 %.1f%% | om 中位=%.3f"%(
        ch,len(s),100*len(s)/len([r for r in rows if r["amp"]>0]),med(amps),pc(amps,.9),
        100*sum(1 for a in amps if a>=1.044)/len(amps), med([r["om"] for r in s])))
print()
print("总有效帧 %d"%len([r for r in rows if r["amp"]>0]))
# 相位源切换频率
prev=None; sw=0
for r in rows:
    if r["amp"]<=0: continue
    if prev is not None and r["ch"]!=prev: sw+=1
    prev=r["ch"]
print("相位源(ch)切换次数 = %d"%sw)
