"""将围棋 Transformer 导出为 MiniZero 的 AlphaZero 推理模型。"""

from copy import deepcopy
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import Tensor, nn

from ...core.model.transformer.transformer import Transformer


class _MiniZeroAlphaZero(nn.Module):
    """提供 MiniZero 加载模型时调用的方法和前向输出。"""

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        self.model = model
        self._board_size = model.board_size
        self._game_name = f"go_{self._board_size}x{self._board_size}"
        self._input_channels = model.stem.conv.in_channels
        self._hidden_channels = model.stem.conv.out_channels
        self._num_blocks = len(model.encoder.blocks)
        self._value_hidden_channels = model.value_head[0].out_features

    @torch.jit.export
    def get_game_name(self) -> str:
        return self._game_name

    @torch.jit.export
    def get_type_name(self) -> str:
        return "alphazero"

    @torch.jit.export
    def get_num_input_channels(self) -> int:
        return self._input_channels

    @torch.jit.export
    def get_input_channel_height(self) -> int:
        return self._board_size

    @torch.jit.export
    def get_input_channel_width(self) -> int:
        return self._board_size

    @torch.jit.export
    def get_num_hidden_channels(self) -> int:
        return self._hidden_channels

    @torch.jit.export
    def get_hidden_channel_height(self) -> int:
        return self._board_size

    @torch.jit.export
    def get_hidden_channel_width(self) -> int:
        return self._board_size

    @torch.jit.export
    def get_num_blocks(self) -> int:
        return self._num_blocks

    @torch.jit.export
    def get_action_size(self) -> int:
        return self._board_size * self._board_size + 1

    @torch.jit.export
    def get_num_value_hidden_channels(self) -> int:
        return self._value_hidden_channels

    @torch.jit.export
    def get_discrete_value_size(self) -> int:
        return 1

    def forward(self, board: Tensor) -> dict[str, Tensor]:
        policy_logit, value = self.model(board)
        black_turn = board[:, 16, 0, 0]
        white_turn = board[:, 17, 0, 0]
        black_value = value * (black_turn - white_turn).unsqueeze(1)
        return {
            "policy_logit": policy_logit.contiguous(),
            "policy": policy_logit.softmax(dim=1),
            "value": black_value.contiguous(),
        }


def export_minizero_torchscript(model: Transformer, path: str | PathLike[str]) -> None:
    """在训练步之间将当前模型权重导出到已准备好的目标目录。"""
    if model.stem.conv.in_channels != 18:
        raise ValueError("MiniZero Go requires 18 input channels.")
    if model.board_size > 19:
        raise ValueError("MiniZero Go board size cannot exceed 19.")

    snapshot = deepcopy(model).to(device="cpu", dtype=torch.float32)
    adapter = _MiniZeroAlphaZero(snapshot).eval().requires_grad_(False)
    scripted = torch.jit.script(adapter)

    destination = Path(path)
    pending = NamedTemporaryFile(
        dir=destination.parent, prefix=".torchscript-", delete=False
    )
    pending_path = Path(pending.name)
    try:
        with pending:
            torch.jit.save(scripted, pending)
        pending_path.replace(destination)
    finally:
        pending_path.unlink(missing_ok=True)
