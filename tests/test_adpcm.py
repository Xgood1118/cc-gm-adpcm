# -*- coding: utf-8 -*-
"""ADPCM 编解码器的规范性与质量测试。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adpcm.core import (  # noqa: E402
    HEADER_SIZE,
    INDEX_TABLE,
    MAX_INDEX,
    STEP_TABLE,
    decode,
    decode_block,
    encode,
    encode_block,
    pack_block,
    read_wav,
    roundtrip,
    snr,
    unpack_block,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_WAV = os.path.join(os.path.dirname(HERE), "data", "source.wav")


def _tone(n, amplitude=12000, freq=220, rate=8000):
    import math
    return [int(amplitude * math.sin(2 * math.pi * freq * i / rate))
            for i in range(n)]


# ---------------------------------------------------------------- 索引更新

def test_index_follows_residual_magnitude():
    # 大残差：索引必须上升；小残差：索引必须下降。
    large = [30000, -30000, 30000, -30000]
    nibbles, _, index_after = encode_block(large, predictor=0, index=20)
    assert index_after > 20
    assert any(INDEX_TABLE[n] > 0 for n in nibbles)

    small = [1, -1, 1, -1, 1, -1]
    nibbles, _, index_after = encode_block(small, predictor=0, index=40)
    assert index_after < 40
    assert all(INDEX_TABLE[n] < 0 for n in nibbles)


def test_index_update_is_bounded():
    # 连续的大幅值交替信号把索引推到顶也不能越界。
    _, _, index_after = encode_block([0, 32767] * 200, index=0)
    assert index_after == MAX_INDEX
    # 零残差持续把索引压到 0，不能变负。
    _, _, index_after = encode_block([0] * 200, index=MAX_INDEX)
    assert index_after == 0


def test_index_update_matches_ima_rule_step_by_step():
    from adpcm.core import _advance
    predictor, index = 0, 10
    nibbles, _, end_index = encode_block([25000, 25000, 25000], predictor, index)
    for nib in nibbles:
        predictor, index = _advance(predictor, index, nib)
    assert index == end_index
    for nib in nibbles:
        assert INDEX_TABLE[nib] != 0


# ------------------------------------------------------------ 4 位残差

def test_residuals_are_four_bits():
    samples = _tone(500)
    nibs, _, _ = encode_block(samples)
    assert all(0 <= n <= 15 for n in nibs)
    # 必须真的用满 4 位：负残差会置符号位，出现 8..15。
    assert any(n > 7 for n in nibs)
    # 也必须出现幅度 3 位的各种取值，不能退化成 3 位量化。
    assert {n & 0x07 for n in nibs} >= {0, 1, 2, 3, 4, 5, 6, 7}


def test_negative_residual_uses_sign_bit_not_three_bit_complement():
    nibs, _, _ = encode_block([-3000], predictor=0, index=30)
    assert nibs[0] & 0x08
    assert nibs[0] & 0x07 <= 7


# ---------------------------------------------------------------- 符号一致

def test_sign_convention_matches_between_encoder_and_decoder():
    for diff, expect_sign in ((3000, 0), (-3000, 0x08)):
        nibs, enc_pred, _ = encode_block([diff], predictor=0, index=25)
        assert (nibs[0] & 0x08) == expect_sign
        out, dec_pred, _ = decode_block(nibs, predictor=0, index=25)
        assert dec_pred == enc_pred
        assert (out[0] > 0) == (diff > 0)


def test_encoder_predictor_and_decoder_step_by_step_aligned():
    samples = [0, 9000, -7000, 12000, -12000, 4000, -4000, 300, -300]
    nibbles, enc_pred, enc_index = encode_block(samples, predictor=0, index=0)
    decoded, dec_pred, dec_index = decode_block(nibbles, predictor=0, index=0)
    assert len(decoded) == len(samples)
    assert (dec_pred, dec_index) == (enc_pred, enc_index)
    # 编码器状态轨迹 = 解码输出轨迹（误差被反馈，不在块内累积）。
    ref_pred, ref_index = 0, 0
    for i, nib in enumerate(nibbles):
        ref_pred, ref_index = decode_block([nib], ref_pred, ref_index)[1:]
        assert decoded[i] == ref_pred


# -------------------------------------------------------------- 块结构与独立解码

def test_block_header_records_initial_state():
    nibs, _, _ = encode_block([100, 200, 300])
    block = pack_block(nibs, predictor=-1234, index=17)
    assert len(block) >= HEADER_SIZE
    got_nibs, got_pred, got_index, count = unpack_block(block)
    assert (got_pred, got_index) == (-1234, 17)
    assert got_nibs == nibs
    assert count == 3


def test_block_bytes_are_byte_aligned_with_padding():
    nibs, pred, idx = encode_block([500, -500, 500, -500, 500])
    block = pack_block(nibs, pred, idx, len(nibs))
    assert (len(block) - HEADER_SIZE) == 3      # 5 nibbles -> 3 bytes
    padded = unpack_block(block)
    assert padded[0] == nibs                    # 填充半字节被 count 截掉


def test_any_block_decodes_independently_from_its_header():
    samples = _tone(37) + [30000, -30000] * 20
    data = encode(samples, block_samples=16)
    blocks = []
    pos = 0
    while pos < len(data):
        count = data[pos + 3] + 1
        size = HEADER_SIZE + (count + 1) // 2
        blocks.append(data[pos:pos + size])
        pos += size
    assert len(blocks) == (len(samples) + 15) // 16

    full = decode(data)
    offset = 0
    for block in blocks:
        nibs, predictor, index, count = unpack_block(block)
        # 只用本块块头起解，不借助上一块的收尾状态。
        part, _, _ = decode_block(nibs, predictor, index, count=count)
        assert part == full[offset:offset + count]
        offset += count


def test_last_block_is_padded_to_whole_bytes():
    samples = _tone(45)                      # 16 + 16 + 13
    data = encode(samples, block_samples=16)
    pos, count_blocks = 0, 0
    while pos < len(data):
        count = data[pos + 3] + 1
        size = HEADER_SIZE + (count + 1) // 2
        assert (size - HEADER_SIZE) == (count + 1) // 2
        pos += size
        count_blocks += 1
    assert count_blocks == 3
    assert len(decode(data)) == len(samples)


def test_roundtrip_preserves_length_and_block_structure():
    samples = _tone(1000)
    rec = roundtrip(samples)
    assert len(rec) == len(samples)
    assert all(-32768 <= v <= 32767 for v in rec)


# ------------------------------------------------------------------ 质量

def test_roundtrip_snr_floor_on_source():
    rate, samples = read_wav(SOURCE_WAV)
    assert rate == 8000 and len(samples) == 2000
    rec = roundtrip(samples)
    total = snr(samples, rec)
    assert total > 20.0, "整体 SNR %.2f dB 低于 20 dB" % total

    half = len(samples) // 2
    first = snr(samples[:half], rec[:half])
    second = snr(samples[half:], rec[half:])
    assert first > 20.0, "前半段 SNR %.2f dB 低于 20 dB" % first
    assert second > 20.0, "后半段 SNR %.2f dB 低于 20 dB" % second
    assert abs(first - second) <= 3.0, "前后半段 SNR 差超过 3 dB"


def test_snr_reports_bad_roundtrip():
    # 反向输出是明显的坏往返，SNR 必须落在负值区间而不是被当成完美重建。
    samples = _tone(200)
    assert snr(samples, [-v for v in samples]) < 0
