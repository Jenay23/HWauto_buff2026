import csv, json
cs=set()
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try: cs.add(int(r["sim_t_ns"]))
        except: pass
ts=set()
for line in open("logs/20261015_141623_fire12.kpts.jsonl"):
    i=line.find(',')
    if i>0:
        try: ts.add(int(line[:i]))
        except: pass
inter=len(cs&ts)
print("CSV 时间戳 %d, kpts 时间戳 %d, 交集 %d (%.1f%%)"%(len(cs),len(ts),inter,100*inter/max(1,len(cs))))
print("=> 交集小说明两边帧对不上，我的时间匹配是错的")
