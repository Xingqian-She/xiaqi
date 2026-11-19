# Transformer Encoder

## EncoderBlock

`EncoderBlock` 是围棋 Transformer Encoder 的单层，由双向多头自注意力和 SwiGLU 组成。 [实现代码](encoder_block.py)

### 接口与形状

- `EncoderBlock(model_dim, num_heads, ffn_hidden_dim, rope)` 接收 token 宽度、注意力头数、SwiGLU 的中间宽度和已构造的 `AxialRoPE2D`。三个数值参数均为正整数。`model_dim` 须能被 `num_heads` 整除，每头宽度 `model_dim // num_heads` 须等于 `rope.head_dim`，并且是 4 的倍数。 [多头自注意力](../attention/README.md) · [SwiGLU](../ffn/README.md)
- `forward(tokens)` 的输入和输出均为 `[B, 1 + n², model_dim]`。序列首位是 CLS，后续棋盘点按行展开。9×9 棋盘对应 82 个 token。
- 更换棋盘尺寸时，用新尺寸构造 `AxialRoPE2D`。保持 token 宽度、头数和 SwiGLU 中间宽度一致，即可用 `load_state_dict` 加载原权重。 [位置编码约定](../positional_encoding/README.md)

下面的示例适用于 `src/model/encoder/` 下按包加载的 Python 模块。宽度和头数为示例值。

```python
import torch

from ..positional_encoding.axial_rope_2d import AxialRoPE2D
from .encoder_block import EncoderBlock

rope = AxialRoPE2D(board_size=9, head_dim=8)
block = EncoderBlock(model_dim=32, num_heads=4, ffn_hidden_dim=48, rope=rope)
tokens = torch.randn(2, 82, 32)
output = block(tokens)
```

### 前向计算

注意力和 SwiGLU 各有一个独立的 `nn.RMSNorm`，均在各自的分支计算前归一化。SwiGLU 读取注意力残差更新后的 token。

```text
x = tokens + attention(attention_norm(tokens))
output = x + ffn(ffn_norm(x))
```

## 多层 Encoder

`Encoder(model_dim, num_heads, ffn_hidden_dim, num_layers, rope)` 按顺序执行 `num_layers` 层 `EncoderBlock`。层数取正整数，各层有独立的可训练参数，共用同一份固定棋盘尺寸的 `AxialRoPE2D`。输入形状为 `[B, 1 + n², model_dim]`，输出是最后一层的 token 序列，形状与输入一致。 [实现代码](encoder.py)

构造时，将每层 `attention.out_proj.weight` 和 `ffn.out_proj.weight` 的初始值乘以 `1/sqrt(2L)`，其中 `L` 是 Encoder 的总层数。更换棋盘尺寸时，按新尺寸和原有频率基数 `theta` 重建 RoPE。保持模型宽度、头数、FFN 中间宽度及层数一致，即可加载原来的 `state_dict`。 [首版实现规划](../../../首版实现规划.md#transformer)

下面演示 `src/model/encoder/` 下按包加载的模块如何调用三层 Encoder。配置参数为示例值。

```python
import torch

from ..positional_encoding.axial_rope_2d import AxialRoPE2D
from .encoder import Encoder

rope = AxialRoPE2D(board_size=9, head_dim=8)
encoder = Encoder(model_dim=32, num_heads=4, ffn_hidden_dim=48, num_layers=3, rope=rope)
tokens = torch.randn(2, 82, 32)
output = encoder(tokens)
```

## 测试

构建好项目的 `dev` 镜像后，从仓库根目录运行下面的命令。

```sh
docker compose run --rm --no-deps --pull never -T dev python -m unittest -v src.model.encoder.encoder_block_test src.model.encoder.encoder_test
```

- 单层测试用参考计算核对两次 Pre-RMSNorm 残差更新，检查梯度、跨尺寸加载权重和 token 数量错误。 [单层测试](encoder_block_test.py)
- 多层测试核对 Block 的执行顺序、独立参数、共享 RoPE 和初始化缩放，并检查非正层数、CPU/CUDA 梯度及 9×9 权重加载到 13×13 后的前向。CUDA 反向用例在 PyTorch 检测到 CUDA 时运行。 [多层测试](encoder_test.py)

在 GPU 开发容器中设置 `XIAQI_TEST_ENCODER_COMPILE=1`，可对比三层 Encoder 在 9×9 输入上的 Inductor 与 eager 输出、输入梯度和参数梯度。

```sh
docker compose run --rm --no-deps --pull never -T -e XIAQI_TEST_ENCODER_COMPILE=1 dev python -m unittest -v src.model.encoder.encoder_test.EncoderTest.test_cuda_compile_matches_eager
```
