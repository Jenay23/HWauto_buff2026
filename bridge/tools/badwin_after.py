import re, statistics
pat = re.compile(r"local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) ch=(\d+) rej=(\d+)(?: heal=(\d+))?")
for f in ["logs/20261014_222230_fps.log","logs/20261014_223458_badwin.log","logs/20261014_224624_fitcheck.log"]:
    rows=[]
    for m in pat.finditer(open(f,errors="ignore").read()):
        rows.append(dict(n=int(m.group(1)),rdy=int(m.group(2)),rms=float(m.group(3)),span=float(m.group(4)),
                         om=float(m.group(7)),A=float(m.group(8)),ch=int(m.group(13)),
                         heal=int(m.group(15)) if m.group(15) else 0))
    if not rows: print(f,"no dual"); continue
    rms=[r["rms"] for r in rows if r["rms"]>=0]
    bad=[r for r in rows if r["rms"]>15]
    # 连续坏窗段长
    runs=[]; s=None
    for i in range(len(rows)):
        if rows[i]["rms"]>15:
            if s is None: s=i
        else:
            if s is not None: runs.append(i-s); s=None
    if s is not None: runs.append(len(rows)-s)
    rdy=100*sum(r["rdy"] for r in rows)/len(rows)
    print("%s"%f)
    print("  rdy=%.0f%% | rms 中位=%.1f p90=%.1f | 坏窗=%.0f%% | 坏窗段: 最長=%d 中位=%.1f 段数=%d | heal+%d"%(
        rdy,statistics.median(rms),sorted(rms)[int(.9*len(rms))],100*len(bad)/len(rows),
        max(runs) if runs else 0, statistics.median(runs) if runs else 0, len(runs),
        rows[-1]["heal"]-rows[0]["heal"]))
    if bad:
        print("     坏窗: om 撞下限 %.0f%% | A 撞上限 %.0f%% | ch分布 %s | n 中位=%.0f"%(
            100*sum(1 for r in bad if r["om"]<=1.885)/len(bad),
            100*sum(1 for r in bad if r["A"]>=1.044)/len(bad),
            {c:sum(1 for r in bad if r["ch"]==c) for c in set(r["ch"] for r in bad)},
            statistics.median([r["n"] for r in bad])))
