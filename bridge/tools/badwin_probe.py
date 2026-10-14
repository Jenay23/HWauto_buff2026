import re, statistics
txt = open("logs/20261014_222230_fps.log", errors="ignore").read()
pat = re.compile(r"\[dual\].*?rcd=([-\d.]+)m.*?\|\s*ctr=(\d+) n=(\d+) rms=([-\d.]+)m\s*\|\s*local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) ch=(\d+) rej=(\d+)")
rows = [dict(rcd=float(m.group(1)), rdym=int(m.group(6)), ln=int(m.group(5)),
             rms=float(m.group(7)), span=float(m.group(8)), w=float(m.group(9)),
             om=float(m.group(11)), A=float(m.group(12)), drop=int(m.group(13)),
             theta=float(m.group(16)), ch=int(m.group(17)), rej=int(m.group(18)))
        for m in pat.finditer(txt)]
print("解析到 [dual] 行:", len(rows))
if rows:
    bad = [r for r in rows if r["rms"] > 15]
    print("坏窗(rms>15°)=%d  占 %.0f%%" % (len(bad), 100*len(bad)/len(rows)))
    good = [r for r in rows if 0 <= r["rms"] <= 15]
    def med(a): a=sorted(a); return a[len(a)//2] if a else float('nan')
    print("\n%-10s %8s %8s %8s %8s %8s %8s" % ("类","n","span","om","A","drop","rej"))
    for tag, s in (("好窗", good), ("坏窗", bad)):
        if not s: continue
        print("%-10s %8.0f %8.0f %8.3f %8.3f %8.0f %8.0f" % (
            tag, med([x["ln"] for x in s]), med([x["span"] for x in s]), med([x["om"] for x in s]),
            med([x["A"] for x in s]), med([x["drop"] for x in s]), med([x["rej"] for x in s])))
    print("\n坏窗 ch 分布:", {c: sum(1 for x in bad if x["ch"]==c) for c in set(x["ch"] for x in bad)})
    print("坏窗 om 分布(min/max): %.3f / %.3f" % (min(x["om"] for x in bad), max(x["om"] for x in bad)))
    print("坏窗 span 分布(min/max): %.0f / %.0f" % (min(x["span"] for x in bad), max(x["span"] for x in bad)))
