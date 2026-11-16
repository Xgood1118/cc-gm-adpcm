# -*- coding: utf-8 -*-
"""IMA ADPCM 编解码骨架。

编码用固定步长索引，解码的残差符号表与规范不一致。
"""

import struct

STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544,
    598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878,
    2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894,
    6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289,
    16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]

INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]


def read_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 12
    rate = None
    samples = []
    while pos < len(data) - 8:
        tag = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        pos += 8 + size
        if tag == b"fmt ":
            rate = struct.unpack("<HHIIHH", body[:16])[2]
        elif tag == b"data":
            n = len(body) // 2
            samples = list(struct.unpack("<%dh" % n, body[:n * 2]))
    return rate, samples


def write_wav(path, rate, samples):
    data = struct.pack("<%dh" % len(samples), *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" +
                b"fmt " + struct.pack("<I", 16) + fmt +
                b"data" + struct.pack("<I", len(data)) + data)


def encode_block(samples, predictor=0, index=0):
    """编码一块。当前步长索引固定不更新，残差只用 3 位。"""
    out = []
    for s in samples:
        step = STEP_TABLE[index]
        diff = s - predictor
        nibble = 0b1000 if diff < 0 else 0
        magnitude = min(7, abs(diff) // max(1, step) )
        nibble |= magnitude
        out.append(nibble)
        predictor += (step * magnitude) if diff >= 0 else -(step * magnitude)
        predictor = max(-32768, min(32767, predictor))
    return out, predictor, index


def decode_block(nibbles, predictor=0, index=0, count=None):
    """解码一块。当前符号判断用 nibble 的最高位，与规范相反。"""
    out = []
    for nib in nibbles:
        step = STEP_TABLE[index]
        magnitude = nib & 0x07
        diff = step * magnitude
        if nib & 0x08:
            diff = -diff
        predictor += diff
        predictor = max(-32768, min(32767, predictor))
        out.append(predictor)
    return out, predictor, index


def roundtrip(samples):
    """整段往返。不做块对齐填充。"""
    block = 32
    result = []
    pred = 0
    idx = 0
    for i in range(0, len(samples), block):
        chunk = samples[i:i + block]
        nibs, pred, idx = encode_block(chunk, pred, idx)
        dec, pred, idx = decode_block(nibs, pred, idx)
        result.extend(dec)
    return result


def snr(orig, rec):
    import math
    n = min(len(orig), len(rec))
    se = sum((orig[i] - rec[i]) ** 2 for i in range(n))
    sig = sum(orig[i] ** 2 for i in range(n))
    if se == 0:
        return 99.0
    return 10 * math.log10(sig / se)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="adpcm")
    ap.add_argument("wav")
    ap.add_argument("--out", default="roundtrip.wav")
    args = ap.parse_args(argv)
    rate, samples = read_wav(args.wav)
    rec = roundtrip(samples)
    write_wav(args.out, rate, rec)
    print("往返完成，%d 采样，SNR %.2f dB" % (len(rec), snr(samples, rec)))
    return 0
