# IMA ADPCM 编解码

骨架能按 32 采样一块做编码与解码，但两块都偏离规范，往返失真很大。

## 已知缺口

- 编码时步长索引从不更新，`INDEX_TABLE` 定义了却没用上，大信号会持续用最小步长，
  量化误差随幅度增长而不收敛。
- 残差的幅度直接用 `abs(diff) // step` 截到 7，没有按规范的四位移位方式取。
- 解码时符号判断写成 `nib & 0x08`，与编码的 `0b1000 if diff < 0` 语义相反，正负互补错位。
- 编码与解码共用同一个 predictor，等于把解码器的状态当编码器状态，误差不反馈。
- 块之间没有 4 字节对齐，也没有块头记录初始 predictor 与索引，解码端无法独立起解。
- 幅度突变处（第 1000 采样之后音量掉到 30%）没有重置策略，误差会持续拖尾。

## 运行

```
python gen_samples.py
python -m adpcm.cli data/source.wav --out roundtrip.wav
python -m pytest tests -q
```

`source.wav` 8k 单声道 2000 采样，220Hz 与 660Hz 叠加、加方波包络，后半段幅度降到 30%。
