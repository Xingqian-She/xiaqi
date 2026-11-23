# 围棋模型模块

模型将棋盘特征转换为落点与 `pass` 的策略 logits，并估计当前执子方的价值。 [模型配置字段](../../data_types/transformer_config/README.md) · [MiniZero 推理导出](../../integrations/minizero/README.md)

| 模块 | 职责 |
| --- | --- |
| [Transformer](transformer/README.md) | 组装卷积 Stem、Encoder、策略头和价值头。 |
| [卷积 Stem](stem/README.md) | 投影输入通道，用残差卷积提取棋盘上的局部特征。 |
| [Encoder](encoder/README.md) | 堆叠带 Pre-RMSNorm 和残差连接的注意力与 FFN 子层。 |
| [多头自注意力](attention/README.md) | 进行双向注意力计算，交换 CLS 与棋盘各点的信息。 |
| [位置编码](positional_encoding/README.md) | 用二维轴向 RoPE 为注意力的 Q 和 K 加入位置信息。 |
| [SwiGLU](ffn/README.md) | 对每个 token 执行门控前馈变换。 |
