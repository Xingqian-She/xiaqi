# SwiGLU

`SwiGLU` 是围棋 Encoder 的前馈分支，用同一组投影权重独立处理每个 token。 [实现代码](swiglu.py)

## 接口与形状

- `SwiGLU(model_dim, hidden_dim)` 接收两个正整数。`model_dim` 是输入和输出宽度，`hidden_dim` 是门控后的中间宽度，由调用方指定。
- `forward(tokens)` 接收形状为 `[..., model_dim]` 的张量，返回同形状的输出。Encoder 中的输入形状为 `[B, 1 + n², model_dim]`，序列由一个 CLS token 和 `n²` 个棋盘 token 组成。切换棋盘尺寸时可以复用同一组 FFN 权重。

下面用 9×9 棋盘对应的 82 个 token 演示调用方式。两个宽度均为示例值。

```python
import torch

from src.core.model.ffn.swiglu import SwiGLU

ffn = SwiGLU(model_dim=32, hidden_dim=48)
tokens = torch.randn(2, 82, 32)
output = ffn(tokens)
```

## 前向计算

1. `gate_up_proj` 将最后一维投影到 `2 * hidden_dim`，沿最后一维均分为 `gate` 和 `up`。
2. 对 `gate` 应用 SiLU，再与 `up` 逐元素相乘。门控结果 `silu(gate) * up` 的宽度为 `hidden_dim`，其中 `silu(gate) = gate * sigmoid(gate)`。
3. `out_proj` 将门控结果投影回 `model_dim`。两层投影均使用 `bias=False`。

## 与 Encoder 的分工

外层 Encoder block 负责 RMSNorm 和残差相加，FFN 子层按 `tokens + ffn(norm(tokens))` 更新。组装 Encoder 时，将 `ffn.out_proj.weight` 在初始化阶段乘以 `1/sqrt(2L)`，其中 `L` 为 Encoder 总层数。 [Encoder 与初始化](../encoder/README.md#多层-encoder)

## 测试

构建项目的 `dev` 镜像后，在仓库根目录运行下面的命令。

```sh
docker compose run --rm --no-deps --pull never -T dev python -m unittest -v src.core.model.ffn.swiglu_test
```

测试用显式公式核对 9×9、13×13 棋盘对应的 token 序列和单个 token 的前向结果，同时检查投影形状和无偏置设置。宽度或输入形状不合法时应抛出 `ValueError`。反向用例检查输入和参数的梯度是否有限且非全零。CUDA 用例在 PyTorch 检测到 CUDA 时运行。 [测试代码](swiglu_test.py)
