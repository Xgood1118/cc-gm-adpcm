# IMA ADPCM 编解码

符合 IMA ADPCM 规范的编解码器，纯 Python 标准库实现。

## 实现要点

- **编码**：每个采样输出 4 位残差（bit3 符号 + bit2..0 幅度）。量化门限与解码端
  位权重重建 `step/8 + b2*step + b1*step/2 + b0*step/4` 完全一致，正负残差用同一
  符号位语义。步长索引按 `INDEX_TABLE` 随残差幅度自适应，并限制在 `0..88`。
- **编码器状态独立**：`encode_block` 只用自身的重建值计算下一步残差，绝不借用解码器
  状态；同一块内编码器与解码器的预测器/索引逐步对齐。
- **块结构**：每块块头 `struct <hBB>` 记录初始预测器(int16)、初始步长索引(uint8) 与
  `采样数-1`。残差每两个打包成一个字节（低半字节在前），末尾不足补 0 半字节，块长
  始终字节对齐。解码只读块头，任意块可独立起解。
- **入口**：`encode`/`decode` 处理整段字节流；`roundtrip(samples)` 编码为字节流后再
  解码，长度与输入一致。
- **已知取舍**：解码器按规范用重建值直接预测下一步（无 LTP 后处理）。因此块起点的
  首个采样没有前置上下文，突变处的单点误差会比块内大 —— 这体现在最大误差上，而不
  影响整体 SNR。

## 质量（`data/source.wav`，2000 采样 @ 8 kHz）

`python -m adpcm.cli data/source.wav --out roundtrip.wav` 实测：

| 指标 | 数值 |
| --- | --- |
| 整体 SNR | 21.25 dB |
| 前半段 SNR | 21.21 dB |
| 后半段 SNR（幅度 30%） | 21.72 dB |
| 前后半段差 | 0.51 dB |
| 最大误差 | 7436 |

## 运行

```
python gen_samples.py
python -m adpcm.cli data/source.wav --out roundtrip.wav
python -m pytest tests -q
```

`source.wav` 8k 单声道 2000 采样，220Hz 与 660Hz 叠加、加方波包络，后半段幅度降到 30%。

## 测试覆盖

`tests/test_adpcm.py` 覆盖：索引随残差幅度双向更新及边界、4 位残差（含符号位与全部
3 位幅度取值）、编解码符号一致与逐步状态对齐、块头独立起解、字节对齐与末尾填充、
`data/source.wav` 上整体/前后半段 SNR 下限及前后差 ≤ 3 dB。
