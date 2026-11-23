# 多头自注意力

`MultiHeadSelfAttention` 是围棋 Encoder 的双向注意力分支。CLS 和所有棋盘 token 可以彼此关注。 [实现代码](multi_head_self_attention.py)

## 接口与形状

- `MultiHeadSelfAttention(model_dim, num_heads, rope)` 接收已构造的 `AxialRoPE2D` 实例。`model_dim` 和 `num_heads` 必须为正，且模型宽度能被头数整除。`head_dim = model_dim // num_heads` 必须与 `rope.head_dim` 一致，且满足二维 RoPE 要求的 4 的倍数。
- `forward(tokens)` 只接收 `[B, T, model_dim]` 的张量，返回同形状的注意力分支输出。`T = 1 + n²`，由 RoPE 的棋盘边长 `n` 决定。序列首位是 CLS，后续棋盘点按行展开；9×9 时共有 82 个 token。
- RoPE 随注意力模块一起迁移设备。切换棋盘尺寸时，用相同的 `head_dim` 为新尺寸构造 RoPE，并提供对应长度的输入；模型宽度和头数不变时，两层投影的权重可以复用。

从仓库根目录使用的示例（宽度和头数仅用于演示）：

```python
import torch

from src.core.model.attention.multi_head_self_attention import MultiHeadSelfAttention
from src.core.model.positional_encoding.axial_rope_2d import AxialRoPE2D

rope = AxialRoPE2D(board_size=9, head_dim=8)
attention = MultiHeadSelfAttention(model_dim=32, num_heads=4, rope=rope)
tokens = torch.randn(2, 82, 32)
output = attention(tokens)
```

## 前向计算

1. `qkv_proj` 用一个无偏置线性层将最后一维从 `model_dim` 投影到 `3 * model_dim`。拆出 Q、K、V 后，每个张量的形状都是 `[B, num_heads, T, head_dim]`。
2. 对 Q 和 K 应用二维轴向 RoPE，V 不旋转。棋盘坐标以中心为原点，CLS 使用固定的 `π` 相位。 [位置编码约定](../positional_encoding/README.md)
3. 调用 PyTorch `scaled_dot_product_attention`，不传注意力掩码，设 `is_causal=False` 和 `dropout_p=0.0`。点积的 `1/sqrt(head_dim)` 缩放由 SDPA 完成，不在模块中重复计算。
4. 合并各头回到 `[B, T, model_dim]`，通过无偏置的 `out_proj` 得到输出。

## 与 Encoder 的分工

外层 `EncoderBlock` 执行 `tokens + attention(norm(tokens))`。组装 Encoder 时，将 `out_proj.weight` 的初值乘以 `1/sqrt(2L)`，其中 `L` 为 Encoder 总层数。前向计算中的点积缩放由 SDPA 处理。 [Encoder 与初始化](../encoder/README.md#多层-encoder)

## 测试

构建好项目的 `dev` 镜像后，在仓库根目录运行：

```sh
docker compose run --rm --no-deps --pull never -T dev python -m unittest -v src.core.model.attention.multi_head_self_attention_test
```

[单元测试](multi_head_self_attention_test.py)在 9×9 输入上用显式计算的双向注意力核对前向结果，检查输出形状和无偏置投影，并验证 CPU/CUDA 上的输出有限、输入和参数都有有限且非全零的梯度。用例也检查不合法的头数、维度和 token 形状。PyTorch 未检测到 CUDA 时跳过 CUDA 用例；这组测试不包含完整 Encoder 或 `torch.compile` 验收。
