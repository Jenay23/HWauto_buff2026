import re
txt = open("logs/20261014_222230_fps.log", errors="ignore").read()
pat = re.compile(r"local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) ch=(\d+) rej=(\d+)")
rows=[dict(n=int(m.group(1)),rdy=int(m.group(2)),rms=float(m.group(3)),span=float(m.group(4)),
           w=float(m.group(5)),om=float(m.group(7)),A=float(m.group(8)),drop=int(m.group(9)),
           theta=float(m.group(12)),ch=int(m.group(13)),rej=int(m.group(14))) for m in pat.finditer(txt)]
# 找一段连续坏窗
bad=[i for i,r in enumerate(rows) if r["rms"]>15]
# 取最长的连续坏窗段
runs=[]; s=None
for i in range(len(rows)):
    if rows[i]["rms"]>15:
        if s is None: s=i
    else:
        if s is not None: runs.append((s,i-1)); s=None
if s is not None: runs.append((s,len(rows)-1))
runs.sort(key=lambda x:x[1]-x[0], reverse=True)
print("坏窗连续段数=%d, 最长=%d 帧"%(len(runs), runs[0][1]-runs[0][0]+1 if runs else 0))
s,e=runs[0]
print("\n最长坏窗段前后 (每行: n rdy rms span w om A drop theta ch rej):")
for i in range(max(0,s-4), min(len(rows), e+5)):
    r=rows[i]; mark="<<< BAD" if r["rms"]>15 else ""
    print("  n=%3d rdy=%d rms=%7.1f span=%4.0f w=%6.2f om=%.3f A=%.3f drop=%3d theta=%7.3f ch=%d rej=%3d %s"%(
        r["n"],r["rdy"],r["rms"],r["span"],r["w"],r["om"],r["A"],r["drop"],r["theta"],r["ch"],r["rej"],mark))
# theta 的帧间变化
print("\n该段 theta 逐行变化(弧度):")
for i in range(max(0,s-2), min(len(rows),e+3)):
    if i>max(0,s-2):
        d=(rows[i]["theta"]-rows[i-1]["theta"]+3.14159)%6.28318-3.14159
        print("   theta差=%+.3f rad (%+.1f deg)  rms=%7.1f  drop=%d"%(d,d*57.2958,rows[i]["rms"],rows[i]["drop"]))
