# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adpcm.core import STEP_TABLE, decode_block, encode_block  # noqa: E402


def test_step_table_bounds():
    assert len(STEP_TABLE) == 89
    assert STEP_TABLE[-1] == 32767


def test_encode_length():
    nibs, pred, idx = encode_block([1000, 2000, 3000])
    assert len(nibs) == 3


def test_decode_length():
    out, pred, idx = decode_block([0x01, 0x02, 0x03])
    assert len(out) == 3
