# -*- coding: utf-8 -*-
"""IMA ADPCM 编解码器。

编码端与解码端使用同一套 IMA 量化/重建规则：

* 每个采样输出 4 位残差（1 位符号 + 3 位幅度）；
* 步长索引按 ``INDEX_TABLE`` 随残差幅度自适应，并限制在 0..88；
* 编码器维护自己的预测器状态，只把重建值写回自身状态，不借用解码器；
* 每块有 4 字节块头（初始预测器、初始步长索引、采样数），块内半字节
  两两打包成字节，末尾不足时补 0 半字节。任意一块都能凭块头独立起解。
"""

import math
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

MAX_INDEX = len(STEP_TABLE) - 1
BLOCK_SAMPLES = 256          # 每块最多 256 个采样
HEADER_SIZE = 4              # <hB B>：predictor(int16) + index(uint8) + (count-1)


def _clamp_predictor(value):
    return -32768 if value < -32768 else (32767 if value > 32767 else value)


def _quantize(diff, step):
    """把残差量化成 4 位半字节（bit3 为符号，bit2..0 为幅度）。

    量化门限与解码端的位权重重建完全一致：
    ``step/8 + bit2*step + bit1*step/2 + bit0*step/4``。
    """
    nibble = 0
    if diff < 0:
        nibble = 0x08
        diff = -diff
    if diff >= step:
        nibble |= 0x04
        diff -= step
    half = step >> 1
    if diff >= half:
        nibble |= 0x02
        diff -= half
    quarter = step >> 2
    if diff >= quarter:
        nibble |= 0x01
    return nibble


def _advance(predictor, index, nibble):
    """按 IMA 规范用半字节更新预测器与步长索引。

    编码端在写入残差后调用它得到自己的重建状态；解码端调用同一个函数，
    因此只要起始状态相同，两边逐步对齐。
    """
    step = STEP_TABLE[index]
    diff = step >> 3
    if nibble & 0x04:
        diff += step
    if nibble & 0x02:
        diff += step >> 1
    if nibble & 0x01:
        diff += step >> 2

    if nibble & 0x08:
        predictor -= diff
    else:
        predictor += diff
    predictor = _clamp_predictor(predictor)

    index += INDEX_TABLE[nibble & 0x0F]
    if index < 0:
        index = 0
    elif index > MAX_INDEX:
        index = MAX_INDEX
    return predictor, index


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
    """编码一块样本。

    返回 ``(nibbles, predictor, index)``。``nibbles`` 每个元素是一个采样
    的 4 位残差；``predictor``/``index`` 是编码器自身的收尾状态。编码器
    只根据自己上一步的重建值计算残差，不读取解码器状态。
    """
    predictor = int(predictor)
    index = max(0, min(MAX_INDEX, int(index)))
    nibbles = []
    for sample in samples:
        sample = int(sample)
        step = STEP_TABLE[index]
        nibble = _quantize(sample - predictor, step)
        nibbles.append(nibble)
        predictor, index = _advance(predictor, index, nibble)
    return nibbles, predictor, index


def decode_block(nibbles, predictor=0, index=0, count=None):
    """解码一块 4 位残差。

    ``count`` 不为 ``None`` 时只解码前 ``count`` 个残差（用于忽略块尾为
    字节对齐补的填充半字节）。返回 ``(samples, predictor, index)``。
    """
    predictor = int(predictor)
    index = max(0, min(MAX_INDEX, int(index)))
    out = []
    for nibble in nibbles:
        if count is not None and len(out) >= count:
            break
        predictor, index = _advance(predictor, index, nibble & 0x0F)
        out.append(predictor)
    return out, predictor, index


def pack_block(nibbles, predictor, index, count=None):
    """把一块残差和块头打包成字节串。

    块头为 ``<hBB``：初始预测器、初始步长索引、``count - 1``。残差每两
    个打包成一个字节（低半字节在前）；块尾不足两个时用 0 半字节填充，
    因此块长总是字节对齐。
    """
    if count is None:
        count = len(nibbles)
    if not 1 <= count <= BLOCK_SAMPLES:
        raise ValueError("块采样数必须在 1..%d 之间" % BLOCK_SAMPLES)
    nibbles = list(nibbles[:count])
    if len(nibbles) != count:
        raise ValueError("残差数量少于声明的块采样数")
    header = struct.pack("<hBB", int(predictor), int(index), count - 1)
    body = bytearray()
    for i in range(0, count, 2):
        low = nibbles[i] & 0x0F
        high = (nibbles[i + 1] & 0x0F) if i + 1 < count else 0
        body.append(low | (high << 4))
    return header + bytes(body)


def unpack_block(data):
    """解析一个块，返回 ``(nibbles, predictor, index, count)``。"""
    if len(data) < HEADER_SIZE:
        raise ValueError("块数据长度不足")
    predictor, index, count_minus_one = struct.unpack("<hBB", data[:HEADER_SIZE])
    count = count_minus_one + 1
    expected = HEADER_SIZE + (count + 1) // 2
    if len(data) != expected:
        raise ValueError("块长度 %d 与块头声明的 %d 不一致" % (len(data), expected))
    if index > MAX_INDEX:
        raise ValueError("块头步长索引越界: %d" % index)
    nibbles = []
    for byte in data[HEADER_SIZE:]:
        nibbles.append(byte & 0x0F)
        nibbles.append((byte >> 4) & 0x0F)
    return nibbles[:count], predictor, index, count


def encode(samples, predictor=0, index=0, block_samples=BLOCK_SAMPLES):
    """把整段采样编码成带块头的 ADPCM 字节流。"""
    if not 1 <= block_samples <= BLOCK_SAMPLES:
        raise ValueError("block_samples 必须在 1..%d 之间" % BLOCK_SAMPLES)
    data = bytearray()
    state_predictor, state_index = int(predictor), int(index)
    for pos in range(0, len(samples), block_samples):
        chunk = samples[pos:pos + block_samples]
        start_predictor, start_index = state_predictor, state_index
        nibbles, state_predictor, state_index = encode_block(
            chunk, start_predictor, start_index)
        data += pack_block(nibbles, start_predictor, start_index, len(chunk))
    return bytes(data)


def decode(data):
    """把 ADPCM 字节流解码成采样列表。

    每个块只使用自己的块头起解，不依赖前一块的收尾状态，因此任意块都能
    独立解码。
    """
    out = []
    pos = 0
    while pos < len(data):
        if pos + HEADER_SIZE > len(data):
            raise ValueError("块头不完整")
        count = data[pos + 3] + 1
        size = HEADER_SIZE + (count + 1) // 2
        block = data[pos:pos + size]
        if len(block) != size:
            raise ValueError("块数据不完整")
        nibbles, predictor, index, count = unpack_block(block)
        samples, _, _ = decode_block(nibbles, predictor, index, count=count)
        out.extend(samples)
        pos += size
    return out


def roundtrip(samples):
    """编码后再解码整段采样。

    解码只读取序列化字节流中的块头，不共享编码器的活状态；返回值长度与
    输入相同。
    """
    return decode(encode(samples))


def snr(orig, rec):
    n = min(len(orig), len(rec))
    se = sum((orig[i] - rec[i]) ** 2 for i in range(n))
    sig = sum(orig[i] ** 2 for i in range(n))
    if se == 0:
        return 99.0
    if sig == 0:
        return -99.0
    return 10 * math.log10(sig / se)


def _max_error(orig, rec):
    n = min(len(orig), len(rec))
    return max(abs(orig[i] - rec[i]) for i in range(n)) if n else 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="adpcm")
    ap.add_argument("wav")
    ap.add_argument("--out", default="roundtrip.wav")
    args = ap.parse_args(argv)
    rate, samples = read_wav(args.wav)
    rec = roundtrip(samples)
    write_wav(args.out, rate, rec)

    half = len(samples) // 2
    whole = snr(samples, rec)
    first = snr(samples[:half], rec[:half])
    second = snr(samples[half:], rec[half:])
    print("往返完成，%d 采样" % len(rec))
    print("整体 SNR: %.2f dB" % whole)
    print("前半段 SNR: %.2f dB" % first)
    print("后半段 SNR: %.2f dB" % second)
    print("最大误差: %d" % _max_error(samples, rec))
    return 0
