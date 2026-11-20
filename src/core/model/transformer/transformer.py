"""围棋 Transformer 的策略与价值网络。"""

import torch
from torch import Tensor, nn

from ..encoder.encoder import Encoder
from ..positional_encoding.axial_rope_2d import AxialRoPE2D
from ..stem.stem import Stem


class Transformer(nn.Module):
    """在固定棋盘尺寸上预测动作分数和当前执子方的价值。"""

    def __init__(
        self,
        board_size: int,
        in_channels: int,
        model_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_layers: int,
        value_hidden_dim: int,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide model_dim")
        if value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be positive")

        self.board_size = board_size
        self.stem = Stem(in_channels, model_dim)
        rope = AxialRoPE2D(board_size, head_dim=model_dim // num_heads)
        self.encoder = Encoder(model_dim, num_heads, ffn_hidden_dim, num_layers, rope)

        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.point_head = nn.Linear(model_dim, 1, bias=False)
        self.pass_head = nn.Linear(model_dim, 1, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, value_hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(value_hidden_dim, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, board: Tensor) -> tuple[Tensor, Tensor]:
        """接收 [B, C, n, n]，返回 [B, n² + 1] 的 logits 和 [B, 1] 的价值。"""
        n = self.board_size
        if board.ndim != 4 or board.size(2) != n or board.size(3) != n:
            raise ValueError(f"board must have shape [batch, channels, {n}, {n}]")

        board_tokens = self.stem(board).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(board.size(0), -1, -1)
        encoded = self.encoder(torch.cat((cls_token, board_tokens), dim=1))
        cls = encoded[:, 0]

        point_logits = self.point_head(encoded[:, 1:]).squeeze(-1)
        pass_logit = self.pass_head(cls)
        policy_logits = torch.cat((point_logits, pass_logit), dim=1)
        return policy_logits, self.value_head(cls)
