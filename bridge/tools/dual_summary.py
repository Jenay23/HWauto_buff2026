#!/usr/bin/env python3
"""解析 [dual] 行汇总。用法: python3 tools/dual_summary.py <log> [<log>...]"""
import re, sys, statistics
pat = re.compile(
    r"local n=(\d+) rdy=(\d+) rms=([-\d.]+)deg span=(\d+)deg w=([-\d.]+) LK=(\d+) "
    r"om=([-\d.]+) A=([-\d.]+) drop=(\d+) lost=(\d+) hit=(\d+) theta=([-\d.]+) "
    r"ch=(\d+) rej=(\d+)(?: heal=(\d+))?")
def med(a):
    a = sorted(a); return a[len(a)//2] if a else float('nan')
def pc(a, q):
    a = sorted(a); return a[min(len(a)-1, int(q*len(a)))] if a else float('nan')
for f in sys.argv[1:]:
    rows = []
    for m in pat.finditer(open(f, errors="ignore").read()):
        rows.append(dict(n=int(m.group(1)), rdy=int(m.group(2)), rms=float(m.group(3)),
                         span=float(m.group(4)), om=float(m.group(7)), A=float(m.group(8)),
                         drop=int(m.group(9)), rej=int(m.group(14)),
                         heal=int(m.group(15)) if m.group(15) else 0))
    if not rows:
        print("%s: no dual" % f); continue
    rms = [r["rms"] for r in rows if r["rms"] >= 0]
    bad = [r for r in rows if r["rms"] > 15]
    print("%s" % f)
    print("  dual=%d  rdy=%.0f%%  | rms 中位=%.1f p90=%.1f max=%.0f | 坏窗(>15°)=%.0f%% | n 中位=%.0f" % (
        len(rows), 100*sum(r["rdy"] for r in rows)/len(rows),
        med(rms), pc(rms, .9), max(rms), 100*len(bad)/len(rows), med([r["n"] for r in rows])))
    if any(r["heal"] for r in rows):
        h = [r["heal"] for r in rows]
        print("  heal: 本次共触发 %d 次" % (h[-1]-h[0]))
    print("  坏窗的 om 分布: 中位=%.3f  撞下限(<=1.885)占 %.0f%% | A 撞上限(>=1.044)占 %.0f%%" % (
        med([r["om"] for r in bad]) if bad else 0,
        100*sum(1 for r in bad if r["om"] <= 1.885)/max(1,len(bad)),
        100*sum(1 for r in bad if r["A"] >= 1.044)/max(1,len(bad))))
