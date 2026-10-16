#!/usr/bin/env python3
"""Reborn27 相机通道探针 —— 判定"慢"在模拟器还是在客户端。

只做协议最小动作：收 16B RBFR 通知 → 读 96B 帧头 → 立刻回 16B RBRA。
不转换、不推理、不拷贝像素（除非加 --copy），所以它测出来的就是
【模拟器真实的发布速率 + 帧龄】。桥跑出 4.6 fps 时用它对照，就能定位。

用法：
    # 先停掉 hw26_sim_bridge —— 模拟器同一时刻只接受一个接收端
    python3 tools/reborn_probe.py                 # 默认测 8 秒
    python3 tools/reborn_probe.py --seconds 15
    python3 tools/reborn_probe.py --copy          # 额外复现桥里那次 4.6MB 像素拷贝

判读：
    fps 很高(>30) 且 age 很小(<50ms)  -> 模拟器没问题，瓶颈在客户端/推理
    fps ≈ 5 且 age ≈ 600ms            -> 瓶颈在模拟器的采集(GPU 渲染+回读)
    fps 很低但 age 很小               -> 模拟器采样太快被丢弃，或 ack 环节慢
"""

import argparse
import mmap
import os
import socket
import statistics
import struct
import sys
import time

HANDSHAKE_LEN = 32
NOTICE_LEN = 16
FRAME_HEADER_LEN = 96


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("模拟器关闭了连接")
        buf += chunk
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--copy", action="store_true",
                    help="每帧额外做一次像素拷贝，复现桥里的成本")
    ap.add_argument("--socket", default="/tmp/reborn27-camera.sock")
    ap.add_argument("--shm", default="/dev/shm/reborn27-camera.ring")
    args = ap.parse_args()

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.socket)
    except OSError as e:
        print(f"[probe] 连不上 {args.socket}: {e}\n"
              f"        模拟器在跑吗？(空 socket 文件的话先删掉再起模拟器)", file=sys.stderr)
        return 2

    hs = recv_exact(sock, HANDSHAKE_LEN)
    if hs[:4] != b"RBIP":
        print(f"[probe] 握手 magic 不对: {hs[:4]!r}", file=sys.stderr)
        return 3
    version, header_len = struct.unpack_from("<HH", hs, 4)
    slots, = struct.unpack_from("<I", hs, 8)
    slot_size, = struct.unpack_from("<Q", hs, 12)
    path_len, = struct.unpack_from("<I", hs, 20)
    shm_path = recv_exact(sock, path_len).decode()
    print(f"[probe] RBIP v{version} header_len={header_len} slots={slots} "
          f"slot_size={slot_size} shm={shm_path}")

    fd = os.open(shm_path, os.O_RDONLY)
    ring = mmap.mmap(fd, slots * slot_size, access=mmap.ACCESS_READ)

    frames = 0
    ages = []
    gaps = []
    seq_list = []
    info = None
    out_of_order = 0
    copy_bytes = 0
    last_seq = None
    last_recv = None
    t_end = time.monotonic() + args.seconds
    t_start = time.monotonic()

    while time.monotonic() < t_end:
        notice = recv_exact(sock, NOTICE_LEN)
        if notice[:4] != b"RBFR":
            print(f"[probe] 通知 magic 不对: {notice[:4]!r}", file=sys.stderr)
            break
        slot, seq = struct.unpack_from("<IQ", notice, 4)
        base = slot * slot_size
        hdr = ring[base:base + FRAME_HEADER_LEN]
        frame_version, frame_header_len = struct.unpack_from("<HH", hdr, 8)
        pixel_format, width, height, stride, pixel_bytes = struct.unpack_from("<IIIII", hdr, 12)
        session, header_seq = struct.unpack_from("<QQ", hdr, 32)
        mono_ns, sim_ns = struct.unpack_from("<QQ", hdr, 48)
        fx, fy, cx, cy = struct.unpack_from("<4d", hdr, 64)

        now_mono = time.monotonic_ns()
        if args.copy:
            blob = ring[base + FRAME_HEADER_LEN: base + FRAME_HEADER_LEN + pixel_bytes]
            copy_bytes += len(blob)

        sock.sendall(struct.pack("<4sIQ", b"RBRA", slot, seq))

        frames += 1
        ages.append((now_mono - mono_ns) / 1e6)
        seq_list.append(seq)
        if last_recv is not None:
            gaps.append((now_mono - last_recv) / 1e6)
        last_recv = now_mono
        if last_seq is not None and seq <= last_seq:
            out_of_order += 1
        last_seq = seq
        if info is None:
            info = (frame_version, frame_header_len, pixel_format, width, height, stride,
                    pixel_bytes, session, sim_ns, fx, fy, cx, cy, header_seq)

    elapsed = time.monotonic() - t_start
    if frames == 0:
        print("[probe] 一帧都没收到", file=sys.stderr)
        return 4

    def stat(name, xs, unit):
        xs = sorted(xs)
        print(f"  {name:22s} min={xs[0]:8.2f}{unit}  p50={statistics.median(xs):8.2f}{unit}  "
              f"p90={xs[int(len(xs) * 0.9)]:8.2f}{unit}  max={xs[-1]:8.2f}{unit}")

    print("\n[probe] === 结果 ===")
    print(f"  时长 {elapsed:.2f}s  收帧 {frames}  平均 {frames / elapsed:.2f} fps")
    stat("帧龄(源->本地)", ages, "ms")
    if gaps:
        stat("相邻收帧间隔", gaps, "ms")
    print(f"  序号乱序 {out_of_order}  序号跨度 {seq_list[-1] - seq_list[0]}")
    if args.copy:
        print(f"  拷贝总量 {copy_bytes / 1e6:.1f} MB "
              f"({copy_bytes / 1e6 / elapsed:.1f} MB/s)")
    if info:
        (fver, fhlen, fmt, w, h, stride, pbytes, ses, sim_ns, fx, fy, cx, cy, hseq) = info
        print(f"  首帧: v{fver} hlen={fhlen} fmt={fmt} {w}x{h} stride={stride} "
              f"pixel_bytes={pbytes} session={ses}")
        print(f"         fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}  "
              f"sim_ns={sim_ns} (按 1e9 换算 sim={sim_ns / 1e9:.1f}s)")

    print("\n[probe] 判读：")
    fps = frames / elapsed
    p50 = statistics.median(ages)
    if fps > 30 and p50 < 50:
        print("  → 模拟器发布很快、帧很新：瓶颈在客户端（转换/推理/邮箱），不在模拟器。")
    elif fps < 15 and p50 > 200:
        print("  → 模拟器发布就慢且帧已经老了：瓶颈在模拟器的采集(渲染+GPU回读)。")
    else:
        print("  → 两者都有份：先看 fps 与 age 各自离目标多远，再决定动哪边。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
