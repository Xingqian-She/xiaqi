"""Stem 的残差卷积分支。"""

from torch import Tensor, nn
from torch.nn.functional import silu


class ResidualConvBlock(nn.Module):
    """计算卷积、逐位置 LayerNorm 和 SiLU，由外层堆叠模块添加残差。"""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")

        self.conv = nn.Conv2d(
            model_dim, model_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, features: Tensor) -> Tensor:
        """接收并返回 [batch, model_dim, height, width] 形状的张量。"""
        if (
            features.ndim != 4
            or features.size(1) != self.conv.in_channels
            or features.size(2) == 0
            or features.size(3) == 0
        ):
            raise ValueError(
                f"features must have shape [batch, {self.conv.in_channels}, height, width] "
                "with height and width > 0"
            )

        convolved = self.conv(features).permute(0, 2, 3, 1)
        return silu(self.norm(convolved)).permute(0, 3, 1, 2)
