# -*- coding: utf-8 -*-
"""生成一段 8k 单声道音频：两个正弦叠一个方波，带幅度突变。"""
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
RATE = 8000
N = 2000


def main():
    samples = []
    for i in range(N):
        t = i / float(RATE)
        v = (12000 * math.sin(2 * math.pi * 220 * t) +
             6000 * math.sin(2 * math.pi * 660 * t) +
             (5000 if (i // 40) % 2 else -5000))
        if i > N // 2:
            v *= 0.3
        samples.append(max(-32768, min(32767, int(v))))
    data = struct.pack("<%dh" % N, *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "source.wav")
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" +
                b"fmt " + struct.pack("<I", 16) + fmt +
                b"data" + struct.pack("<I", len(data)) + data)
    print("已生成", path, N, "采样")


if __name__ == "__main__":
    main()
