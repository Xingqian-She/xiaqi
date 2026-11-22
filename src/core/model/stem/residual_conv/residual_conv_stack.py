"""按顺序组装 Stem 的残差卷积块。"""

from numbers import Integral

from torch import Tensor, nn

from .residual_conv_block import ResidualConvBlock


class ResidualConvStack(nn.Module):
    """依次应用独立的卷积分支，并在每一层加上该层输入。"""

    def __init__(self, model_dim: int, num_blocks: int) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if (
            not isinstance(num_blocks, Integral)
            or isinstance(num_blocks, bool)
            or num_blocks < 0
        ):
            raise ValueError("num_blocks must be a nonnegative integer")

        self.blocks = nn.ModuleList(
            ResidualConvBlock(model_dim) for _ in range(num_blocks)
        )

    def forward(self, features: Tensor) -> Tensor:
        """更新 [batch, model_dim, height, width] 的特征，保持形状。"""
        for block in self.blocks:
            features = features + block(features)
        return features
