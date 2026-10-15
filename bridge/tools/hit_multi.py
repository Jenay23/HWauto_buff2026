import csv
for f in ["logs/20261015_195559_fire12.rtstat.csv","logs/20261015_200727_fire12.rtstat.csv"]:
    rows=[(float(r["wall_s"]),int(r["launch"]),int(r["hits"])) for r in csv.DictReader(open(f))]
    print("=== %s ==="%f.split("/")[-1])
    odd=0
    for i in range(1,len(rows)):
        dl=rows[i][1]-rows[i-1][1]; dh=rows[i][2]-rows[i-1][2]
        if dh>dl:
            odd+=1
            if odd<=6: print("  t=%.1f: 发弹 +%d 但命中 +%d   <- 一发多中?"%(rows[i][0]-rows[0][0],dl,dh))
        if dl>0 and dh>0 and dh>dl+1:
            pass
    print("  累计: %d 次'命中增量 > 发弹增量'"%odd)
    # hits 单调性
    hs=[r[2] for r in rows]
    print("  hits 单调:", all(hs[i]>=hs[i-1] for i in range(1,len(hs))))
