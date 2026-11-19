"""以棋盘中心为原点的二维轴向旋转位置编码。"""

import math

import torch
from torch import Tensor, nn


class AxialRoPE2D(nn.Module):
    """旋转形状为 [batch, heads, 1 + board_size**2, head_dim] 的 Q 和 K。

    第一个 token 是 CLS，后续棋盘点按行展开。每个头的前半通道使用列坐标 x，
    后半通道使用行坐标 y，每两个相邻通道组成一个旋转平面。
    """

    def __init__(self, board_size: int, head_dim: int, theta: float = 100.0) -> None:
        """创建固定棋盘尺寸的旋转表。

        board_size 是棋盘边长，取正奇数。
        head_dim 是每个注意力头的维度，取 4 的正倍数。
        theta 是频率基数，增大它会降低每轴从第二对起的旋转频率。
        """
        super().__init__()
        if board_size <= 0 or board_size % 2 == 0:
            raise ValueError("board_size must be a positive odd integer")
        if head_dim <= 0 or head_dim % 4 != 0:
            raise ValueError("head_dim must be a positive multiple of 4")
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive")

        self.num_tokens = 1 + board_size**2
        self.head_dim = head_dim
        axis = torch.arange(board_size, dtype=torch.get_default_dtype()) - board_size // 2
        y, x = torch.meshgrid(axis, axis, indexing="ij")

        pairs_per_axis = head_dim // 4
        pair_index = torch.arange(pairs_per_axis, dtype=axis.dtype)
        frequencies = theta ** (-pair_index / pairs_per_axis)
        x_angles = torch.outer(x.flatten(), frequencies)
        y_angles = torch.outer(y.flatten(), frequencies)
        board_angles = torch.cat((x_angles, y_angles), dim=1)
        # CLS 每对通道固定转半圈。每轴首对的频率为 1，整数棋盘坐标
        # 对应的相位在模 2π 下均与 π 不同。
        cls_angles = torch.full_like(board_angles[:1], math.pi)
        angles = torch.cat((cls_angles, board_angles), dim=0)

        # 角度表由配置生成，随模块移动，切换棋盘尺寸时可重新生成。
        self.register_buffer("cos_angles", angles.cos(), persistent=False)
        self.register_buffer("sin_angles", angles.sin(), persistent=False)

    def _rotate(self, values: Tensor) -> Tensor:
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated_even = even * self.cos_angles - odd * self.sin_angles
        rotated_odd = even * self.sin_angles + odd * self.cos_angles
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.flatten(-2).to(dtype=values.dtype)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        if query.ndim != 4 or query.shape != key.shape:
            raise ValueError("query and key must have matching [batch, heads, tokens, head_dim] shapes")
        if query.size(2) != self.num_tokens or query.size(3) != self.head_dim:
            raise ValueError(f"expected {self.num_tokens} tokens and head_dim={self.head_dim}")
        if not query.is_floating_point() or not key.is_floating_point():
            raise ValueError("query and key must be floating-point tensors")
        return self._rotate(query), self._rotate(key)
