"""应用二维轴向 RoPE 的多头自注意力。"""

from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from ..positional_encoding.axial_rope_2d import AxialRoPE2D


class MultiHeadSelfAttention(nn.Module):
    """双向多头自注意力，只计算注意力分支。

    本模块不执行 RMSNorm，也不添加残差连接。
    out_proj 的初始化缩放、归一化和残差由外层 Encoder block 负责。
    """

    def __init__(self, model_dim: int, num_heads: int, rope: AxialRoPE2D) -> None:
        super().__init__()
        if model_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.head_dim = model_dim // num_heads
        if rope.head_dim != self.head_dim:
            raise ValueError(f"expected rope.head_dim={self.head_dim}, got {rope.head_dim}")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.rope = rope
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """接收并返回形状为 [batch, 1 + board_size**2, model_dim] 的张量。

        第一个 token 是 CLS，后续棋盘点按行展开。RoPE 仅作用于 Q 和 K。
        """
        if tokens.ndim != 3 or tokens.size(-1) != self.model_dim:
            raise ValueError(f"tokens must have shape [batch, tokens, {self.model_dim}]")

        batch_size, num_tokens, _ = tokens.shape
        qkv = self.qkv_proj(tokens).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.transpose(1, 3).unbind(dim=2)
        query, key = self.rope(query, key)
        attended = scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        merged = attended.transpose(1, 2).reshape(batch_size, num_tokens, self.model_dim)
        return self.out_proj(merged)
