import re
pat = re.compile(r"droll=\(([-\d.]+),([-\d.]+)\)deg pick=\((\d),(\d)\) dres=([-\d.]+)deg chord=([-\d.]+)m rcd=([-\d.]+)m dlt=([-\d.]+).*?local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) ch=(\d+) rej=(\d+)(?: heal=(\d+))?")
for f in ["logs/20261015_195559_fire12.log","logs/20261015_200727_fire12.log"]:
    rows=[]
    for m in pat.finditer(open(f,errors="ignore").read()):
        rows.append(dict(dlt=float(m.group(8)), n=int(m.group(9)), rdy=int(m.group(10)), rms=float(m.group(11)),
                         span=float(m.group(12)), om=float(m.group(15)), A=float(m.group(16)),
                         drop=int(m.group(17)), theta=float(m.group(20)), ch=int(m.group(21)),
                         rej=int(m.group(22)), heal=int(m.group(23)) if m.group(23) else 0))
    print("=== %s  (%d 行) ==="%(f.split("/")[-1],len(rows)))
    import collections
    print("  dlt 分布:", dict(collections.Counter(round(r["dlt"]) for r in rows)))
    print("  ch 分布:", dict(collections.Counter(r["ch"] for r in rows)))
    print("  om 撞边界数: 下限=%d 上限=%d | A 撞边界数: 下限=%d 上限=%d"%(
        sum(1 for r in rows if r["om"]<=1.8855), sum(1 for r in rows if r["om"]>=1.9995),
        sum(1 for r in rows if r["A"]<=0.7805), sum(1 for r in rows if r["A"]>=1.0445)))
    # 连续 rdy=0 段
    runs=[];s=None
    for i,r in enumerate(rows):
        if r["rdy"]==0:
            if s is None: s=i
        else:
            if s is not None: runs.append(i-s); s=None
    if s is not None: runs.append(len(rows)-s)
    runs.sort(reverse=True)
    print("  rdy=0 连续段: 最長=%d 段数=%d (前5长: %s)"%(runs[0] if runs else 0,len(runs),runs[:5]))
    # dlt 突变
    sw=[i for i in range(1,len(rows)) if abs(rows[i]["dlt"]-rows[i-1]["dlt"])>20]
    print("  dlt 突变次数=%d"%len(sw))
    # theta 突变
    import math
    def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi
    big=[i for i in range(1,len(rows)) if abs(wrap(rows[i]["theta"]-rows[i-1]["theta"]))>1.5]
    print("  theta 单步>1.5rad 次数=%d"%len(big))
