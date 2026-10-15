import csv, json, bisect
FX=FY=1738.5; CX=720.0; CY=540.0
cs=[]
with open("hw26_bridge_log.csv") as f:
    for r in csv.DictReader(f):
        try: cs.append((int(r["sim_t_ns"]), float(r["aim_x"]), float(r["aim_y"]), int(r["control"]), float(r["lead_s"])))
        except: pass
lit={}
for line in open("logs/20261015_141623_fire12.kpts.jsonl"):
    i=line.find(',')
    if i<0: continue
    try: rec=json.loads(line[i+1:])
    except: continue
    s=rec.get("sim")
    if not s: continue
    p=[]
    for t in s["targets"]:
        if t["state"]==1:
            c=t["center_cam"]; z=c[2]
            if z>1e-6: p.append((FX*c[0]/z+CX, FY*c[1]/z+CY))
    lit[int(line[:i])]=p
n=0
for ts,ax,ay,c,lead in cs:
    if c!=1 or ax<0: continue
    if ts not in lit: continue
    print("t=%d aim=(%.0f,%.0f) lead=%.2f | 当前亮靶标投影:"%(ts,ax,ay,lead), [(round(x),round(y)) for x,y in lit[ts]])
    n+=1
    if n>=8: break
