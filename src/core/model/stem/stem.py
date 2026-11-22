"""投影棋盘输入并提取围棋 Transformer 的局部特征。"""

from torch import Tensor, nn

from .residual_conv.residual_conv_stack import ResidualConvStack


class Stem(nn.Module):
    """以保持空间尺寸的卷积为每个棋盘交叉点生成特征。"""

    def __init__(self, in_channels: int, model_dim: int, num_blocks: int = 1) -> None:
        super().__init__()
        if in_channels <= 0 or model_dim <= 0:
            raise ValueError("in_channels and model_dim must be positive")

        self.conv = nn.Conv2d(
            in_channels, model_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.residual_stack = ResidualConvStack(model_dim, num_blocks)

    def forward(self, board: Tensor) -> Tensor:
        """将 [batch, in_channels, n, n] 投影为 [batch, model_dim, n, n]。"""
        if (
            board.ndim != 4
            or board.size(1) != self.conv.in_channels
            or board.size(2) != board.size(3)
            or board.size(2) == 0
        ):
            raise ValueError(
                f"board must have shape [batch, {self.conv.in_channels}, n, n] with n > 0"
            )

        return self.residual_stack(self.conv(board))
