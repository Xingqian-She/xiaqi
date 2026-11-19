# Stem

`Stem` 为围棋 Transformer 投影棋盘特征，每个交叉点得到一个 `model_dim` 维向量。 [实现代码](stem.py)

## 接口与形状

- `Stem(in_channels, model_dim)` 接收两个正整数，分别对应棋盘输入通道数 `C` 和模型宽度 `d`。
- `forward(board)` 接收浮点张量 `[B, C, n, n]`，返回 `[B, d, n, n]`。`B` 为批大小，棋盘边长 `n` 由输入决定。同一个 Stem 可以处理 9×9、13×13 等尺寸的棋盘。
- 前向计算使用一层带零填充的 `3×3 Conv2d`，设置 `stride=1`、`padding=1` 和 `bias=False`。

Transformer 将 Stem 的输出按行展平为 `[B, n², d]`，在序列开头拼接 CLS 后送入 Encoder。 [首版实现规划](../../../首版实现规划.md#transformer)

下面用 9×9 的合成输入演示调用方式。通道数和模型宽度均为示例值。

```python
import torch

from src.model.stem.stem import Stem

stem = Stem(in_channels=3, model_dim=32)
board = torch.randn(2, 3, 9, 9)
features = stem(board)  # [2, 32, 9, 9]
```

## 测试

构建项目的 `dev` 镜像后，从仓库根目录运行下面的命令。

```sh
docker compose run --rm --no-deps --pull never -T dev python -m unittest -v src.model.stem.stem_test
```

测试覆盖同一实例的 9×9 和 13×13 形状、边界零填充、无偏置、输入和权重的梯度，以及不合法的构造参数和输入形状。CUDA 反向用例在 PyTorch 检测到 CUDA 时运行。 [测试代码](stem_test.py)
