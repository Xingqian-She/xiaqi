"""围棋 Transformer Encoder 的单层模块。"""

from torch import Tensor, nn

from ..attention.multi_head_self_attention import MultiHeadSelfAttention
from ..ffn.swiglu import SwiGLU
from ..positional_encoding.axial_rope_2d import AxialRoPE2D


class EncoderBlock(nn.Module):
    """依次执行 Pre-RMSNorm 自注意力和 SwiGLU 残差更新。

    rope 提供固定棋盘尺寸的旋转表。组装多层 Encoder 时，按总层数
    缩放 attention.out_proj 和 ffn.out_proj 的初始权重。
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        rope: AxialRoPE2D,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(model_dim, num_heads, rope)
        self.attention_norm = nn.RMSNorm(model_dim)
        self.ffn = SwiGLU(model_dim, ffn_hidden_dim)
        self.ffn_norm = nn.RMSNorm(model_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """接收并返回 [batch, 1 + board_size**2, model_dim] 形状的张量。

        序列首位是 CLS，后续棋盘点按行展开。
        """
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.ffn(self.ffn_norm(tokens))
