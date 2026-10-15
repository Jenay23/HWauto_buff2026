import re
pat = re.compile(r"local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) ch=(\d+) rej=(\d+) heal=(\d+)")
for f in ["logs/20261015_200727_fire12.log","logs/20261015_195559_fire12.log"]:
    rows=[]
    for m in pat.finditer(open(f,errors="ignore").read()):
        rows.append(dict(n=int(m.group(1)),rdy=int(m.group(2)),rms=float(m.group(3)),span=float(m.group(4)),
                         om=float(m.group(7)),A=float(m.group(8)),drop=int(m.group(9)),rej=int(m.group(14)),
                         heal=int(m.group(15))))
    print("=== %s (%d 条) ==="%(f.split("/")[-1],len(rows)))
    # rej 冻结的起点
    frozen=[i for i in range(1,len(rows)) if rows[i]["rej"]==rows[i-1]["rej"]]
    last_change=max([i for i in range(1,len(rows)) if rows[i]["rej"]!=rows[i-1]["rej"]], default=0)
    print("  rej 序列:", [r["rej"] for r in rows][-20:])
    print("  rej 最后一次变化在第 %d 条（共 %d 条）-> 之后冻结了 %d 条"%(last_change,len(rows),len(rows)-last_change))
    # 卡死状态: rdy=0 但 rms<15
    stuck=[i for i,r in enumerate(rows) if r["rdy"]==0 and 0<=r["rms"]<15]
    print("  ★ 'rms<15° 却 rdy=0' 的条数 = %d (%.0f%%)"%(len(stuck),100*len(stuck)/len(rows)))
    print("     最后 12 条的 (span, rms, rdy):", [(r["span"],round(r["rms"],1),r["rdy"]) for r in rows[-12:]])
