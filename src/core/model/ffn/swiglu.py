"""Transformer Encoder 的 SwiGLU 前馈网络。"""

from torch import Tensor, nn
from torch.nn.functional import silu


class SwiGLU(nn.Module):
    """逐 token 计算的前馈分支，hidden_dim 指定门控后的中间宽度。

    外层 Encoder 负责 RMSNorm、残差相加和 out_proj 的初始化缩放。
    """

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if model_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model_dim and hidden_dim must be positive")

        self.model_dim = model_dim
        self.gate_up_proj = nn.Linear(model_dim, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """接收形状为 [..., model_dim] 的张量，返回同形状的输出。"""
        if tokens.ndim == 0 or tokens.size(-1) != self.model_dim:
            raise ValueError(f"tokens must have shape [..., {self.model_dim}]")

        gate, up = self.gate_up_proj(tokens).chunk(2, dim=-1)
        return self.out_proj(silu(gate) * up)
