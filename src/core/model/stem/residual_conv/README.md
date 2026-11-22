# 残差卷积

`ResidualConvBlock(model_dim)` 接收形状为 `[B, d, H, W]` 的特征，输出同形状的 `SiLU(LayerNorm(Conv3×3(x)))`。卷积保持通道数 `d` 和空间尺寸，使用 `stride=1`、`padding=1`、`bias=False`。LayerNorm 在每个位置独立对 `d` 个通道做归一化，使用同一组可学习的仿射参数。 [分支实现](residual_conv_block.py)

`ResidualConvStack(model_dim, num_blocks)` 依次应用 `num_blocks` 个独立的卷积分支。每层执行 `x = x + block(x)`，并将结果送入下一层。`num_blocks` 取非负整数，设为 `0` 时直接返回输入。 [堆叠实现](residual_conv_stack.py) · [Stem](../README.md)

## 测试

在有 PyTorch 的环境中，从仓库根目录运行。

```sh
python -m unittest -v src.core.model.stem.residual_conv.residual_conv_block_test src.core.model.stem.residual_conv.residual_conv_stack_test
```

测试用显式公式核对卷积和逐位置 LayerNorm，并覆盖分支梯度、多层残差更新、零层及无效输入。 [分支测试](residual_conv_block_test.py) · [堆叠测试](residual_conv_stack_test.py)
