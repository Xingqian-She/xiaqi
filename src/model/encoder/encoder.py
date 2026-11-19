"""由多层 Pre-RMSNorm Block 组成的围棋 Transformer Encoder。"""

import math

import torch
from torch import Tensor, nn

from ..positional_encoding.axial_rope_2d import AxialRoPE2D
from .encoder_block import EncoderBlock


class Encoder(nn.Module):
    """依次执行多层 EncoderBlock，各层共用固定棋盘尺寸的 RoPE。"""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_layers: int,
        rope: AxialRoPE2D,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.blocks = nn.ModuleList(
            EncoderBlock(model_dim, num_heads, ffn_hidden_dim, rope)
            for _ in range(num_layers)
        )

        # 残差分支的输出投影在构造时按总层数缩放一次。
        residual_scale = 1.0 / math.sqrt(2 * num_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attention.out_proj.weight.mul_(residual_scale)
                block.ffn.out_proj.weight.mul_(residual_scale)

    def forward(self, tokens: Tensor) -> Tensor:
        """返回最后一层的 token，保持 [batch, 1 + board_size**2, model_dim]。"""
        for block in self.blocks:
            tokens = block(tokens)
        return tokens
