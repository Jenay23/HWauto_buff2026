import csv, math, statistics
# 1) 理论灵敏度：d=7m, v0=22, g=9.81，h 从 -1 到 +1
def fly(d,h,v0=22.0,g=9.81):
    v2=v0*v0; disc=v2*v2-g*(g*d*d+2.0*h*v2)
    if disc<0: return None
    root=math.sqrt(disc); tan=(v2-root)/(g*d); th=math.atan(tan)
    return d/(v0*math.cos(th)), math.degrees(th)
print("理论：d=7.0 m, v0=22 m/s")
print("  %8s %10s %10s"%("h(m)","fly(s)","pitch(deg)"))
for h in (-1.0,-0.5,0,0.5,1.0,1.5,2.0):
    r=fly(7.0,h)
    if r: print("  %8.1f %10.4f %10.2f"%(h,r[0],r[1]))
# 2) 实测 fly_s 变化
f=[]
with open("HWauto_buff2026/bridge/hw26_bridge_log.csv") as fh:
    for row in csv.DictReader(fh):
        try:
            v=float(row["fly_s"])
            if v>0: f.append(v)
        except: pass
f.sort(); n=len(f)
print()
print("实测 fly_s: p10=%.4f p50=%.4f p90=%.4f min=%.4f max=%.4f 极差=%.1f ms"%(
    f[int(.1*n)],f[n//2],f[int(.9*n)],f[0],f[-1],1000*(f[-1]-f[0])))
