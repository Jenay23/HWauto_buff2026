import re
pat = re.compile(r"local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) ch=(\d+) rej=(\d+) heal=(\d+)")
rows=[dict(n=int(m.group(1)),rdy=int(m.group(2)),rms=float(m.group(3)),span=float(m.group(4)),
           w=float(m.group(5)),om=float(m.group(7)),A=float(m.group(8)),drop=int(m.group(9)),
           theta=float(m.group(12)),rej=int(m.group(14)),heal=int(m.group(15)))
      for m in pat.finditer(open("logs/20261014_224624_fitcheck.log",errors="ignore").read())]
print("总 dual 行:",len(rows))
# 找 heal 突增的位置（刚触发的下一帧）
print("\nheal 触发后的几帧（前 6 次）:")
cnt=0
for i in range(1,len(rows)):
    if rows[i]["heal"]>rows[i-1]["heal"] and cnt<6:
        cnt+=1
        print("--- 触发 #%d (heal %d->%d) ---"%(cnt,rows[i-1]["heal"],rows[i]["heal"]))
        for j in range(max(0,i-2), min(len(rows), i+3)):
            r=rows[j]
            print("   n=%3d rdy=%d rms=%7.1f span=%4.0f om=%.3f A=%.3f drop=%3d rej=%3d theta=%7.3f heal=%d %s"%(
                r["n"],r["rdy"],r["rms"],r["span"],r["om"],r["A"],r["drop"],r["rej"],r["theta"],r["heal"],
                "<<<BAD" if r["rms"]>15 else ""))
