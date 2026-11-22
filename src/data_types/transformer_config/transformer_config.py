from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """围棋 Transformer 的棋盘尺寸和网络结构参数。"""

    board_size: int
    in_channels: int
    model_dim: int
    num_heads: int
    ffn_hidden_dim: int
    num_layers: int
    value_hidden_dim: int
    num_stem_blocks: int = 1
