# 围棋 Transformer

`Transformer` 由卷积 Stem、多层 Encoder 和策略与价值输出头组成。 [实现代码](transformer.py) · [首版实现规划](../../../首版实现规划.md#transformer)

## 接口与形状

- 构造时传入棋盘边长 `board_size`、输入通道数 `in_channels`、模型宽度 `model_dim`、注意力头数 `num_heads`、FFN 中间宽度 `ffn_hidden_dim`、Encoder 层数 `num_layers` 和价值头中间宽度 `value_hidden_dim`。棋盘边长取正奇数，`model_dim` 须能被 `num_heads` 整除，每头宽度须是 4 的倍数。
- `forward(board)` 接收形状为 `[B, C, n, n]` 的浮点张量，其中 `C = in_channels`，`n = board_size`。返回 `(policy_logits, value)`，形状分别为 `[B, n² + 1]` 和 `[B, 1]`。
- 策略输出是原始 logit。棋盘点 `(r, c)` 对应下标 `r * n + c`，`pass` 对应下标 `n²`。价值取值为 `[-1, 1]`，估计当前执子方的终局结果。

下面用 9×9 的合成输入演示接口。构造参数均为示例值。

```python
import torch

from src.model.transformer.transformer import Transformer

model = Transformer(
    board_size=9,
    in_channels=3,
    model_dim=32,
    num_heads=4,
    ffn_hidden_dim=48,
    num_layers=3,
    value_hidden_dim=16,
)
board = torch.randn(2, 3, 9, 9)
policy_logits, value = model(board)  # [2, 82], [2, 1]
```

## 前向计算

Stem 输出按行展开为棋盘 token。Transformer 在序列开头拼接一个可学习的 CLS 向量，并用均值为 0、标准差为 0.02 的正态分布初始化该向量。Encoder 以固定棋盘尺寸生成的二维轴向 RoPE 处理序列。 [Stem](../stem/README.md) · [Encoder](../encoder/README.md)

最后一层的棋盘 token 共用一层 `Linear(d, 1)` 生成落点分数。CLS 经独立的 `Linear(d, 1)` 生成 `pass` 分数，经 `Linear(d, d_v) → SiLU → Linear(d_v, 1) → tanh` 生成价值。各输出头的线性层均使用 `bias=False`。

切换棋盘尺寸时，用新边长和相同的其余参数构造模型，再用 `load_state_dict` 加载原权重。新实例会生成对应尺寸的 RoPE 表。

## 测试

构建项目的 `dev` 镜像后，从仓库根目录运行下面的命令。

```sh
docker compose run --rm --no-deps --pull never -T dev python -m unittest -v src.model.transformer.transformer_test
```

测试核对落点和 `pass` 的顺序、价值头计算、无偏置设置、CPU/CUDA 梯度，以及将 9×9 权重载入 13×13 模型后的输出。CUDA 反向用例在 PyTorch 检测到 CUDA 时运行。 [测试代码](transformer_test.py)

在 GPU 开发容器中设置 `XIAQI_TEST_TRANSFORMER_COMPILE=1`，可对比整网在 9×9 输入上的 Inductor 与 eager 输出、输入梯度和参数梯度。

```sh
docker compose run --rm --no-deps --pull never -T -e XIAQI_TEST_TRANSFORMER_COMPILE=1 dev python -m unittest -v src.model.transformer.transformer_test.TransformerTest.test_cuda_compile_matches_eager
```
