# TransformerConfig

`TransformerConfig` 保存棋盘尺寸和网络结构参数。将实例传给 `Transformer.from_config(config)` 构建模型，模型构造时检查配置值。 [类型定义](transformer_config.py) · [构造示例](../../core/model/transformer/README.md#接口与形状)

| 字段 | 含义 |
| --- | --- |
| `board_size` | 固定棋盘边长，取正奇数。 |
| `in_channels` | 棋盘输入通道数，MiniZero Go 使用 `18`。 |
| `model_dim` | Stem 输出及 Encoder token 的通道宽度。 |
| `num_heads` | 注意力头数，须整除 `model_dim`。每头宽度须是 4 的倍数。 |
| `ffn_hidden_dim` | SwiGLU 的中间宽度。 |
| `num_layers` | Encoder 层数。 |
| `value_hidden_dim` | 价值头的中间宽度。 |
| `num_stem_blocks` | Stem 残差卷积块数，取非负整数，默认 `1`。设为 `0` 可加载原单卷积 Stem 的模型权重。 |
