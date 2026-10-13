#!/usr/bin/env python3
# 从 POWER.glb 导出 FACE_1/FACE_2 每个靶标 8 个真值点在【符文坐标系】里的偏移。
import struct, json, numpy as np, sys
p = sys.argv[1] if len(sys.argv) > 1 else "at_vision_simulator/src/assets/POWER.glb"
data = open(p, "rb").read(); length = struct.unpack("<I", data[8:12])[0]
off = 12; js = None
while off < length:
    clen, ctype = struct.unpack("<II", data[off:off+8])
    if ctype == 0x4E4F534A: js = json.loads(data[off+8:off+8+clen])
    off += 8 + clen
nodes = js["nodes"]
loc8 = np.array([[0.09088,-0.07531,-0.09432],[0.10103,-0.04152,-0.10446],
                 [0.05215,0.12956,-0.05558],[0.02825,0.14391,-0.03169],
                 [-0.09272,0.07479,0.08928],[-0.10286,0.04099,0.09943],
                 [-0.05398,-0.13009,0.05055],[-0.03009,-0.14443,0.02666]])
def T4(n):
    t = np.array(n.get("translation",[0,0,0]), np.float64)
    q = np.array(n.get("rotation",[0,0,0,1]), np.float64); x,y,z,w = q
    R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    R = R * np.array(n.get("scale",[1,1,1]), np.float64)[None,:]
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = t; return M
parent = {}
for i,n in enumerate(nodes):
    for c in n.get("children",[]): parent[c] = i
world = {}
def W(i):
    if i in world: return world[i]
    M = T4(nodes[i])
    if i in parent: M = W(parent[i]) @ M
    world[i] = M; return M
for face in (1,2):
    fi = [i for i,n in enumerate(nodes) if n.get("name") == "FACE_%d" % face][0]
    Wi = np.linalg.inv(W(fi)); tab = []
    for t in range(1,6):
        idx = [i for i,n in enumerate(nodes) if n.get("name") == "FACE_%d_TARGET_%d_ACTIVE" % (face,t)][0]
        M = Wi @ W(idx); tab.append((M[:3,:3] @ loc8.T).T + M[:3,3])
    tab = np.array(tab)
    name = "FACE%d_8PTS_RUNE" % face
    print("pub const %s: [[[f32; 3]; 8]; 5] = [" % name)
    for t in range(5):
        print("    [")
        for i in range(8):
            print("        [%.6f, %.6f, %.6f]," % (tab[t,i,0], tab[t,i,1], tab[t,i,2]))
        print("    ],")
    print("];")
