"""保存与恢复 Learner 的模型、优化器、步数和可选的 EMA 权重。"""

from collections.abc import Callable
from copy import deepcopy
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    ema_model: nn.Module | None = None,
) -> None:
    """保存训练状态，写入完成后替换目标文件。"""
    destination = Path(path)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()
    pending = NamedTemporaryFile(
        dir=destination.parent, prefix=".checkpoint-", delete=False
    )
    pending_path = Path(pending.name)
    try:
        with pending:
            torch.save(state, pending)
        pending_path.replace(destination)
    finally:
        pending_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    normalize_optimizer_state: Callable[[dict[str, Any]], dict[str, Any]],
    ema_model: nn.Module | None = None,
) -> int:
    """就地恢复模型、优化器和可选的 EMA 权重，成功后返回训练步数。"""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model_state = state["model"]
    optimizer_state = state["optimizer"]
    step = state["step"]
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer.")
    optimizer_state = normalize_optimizer_state(optimizer_state)

    previous_model = deepcopy(model.state_dict())
    previous_optimizer = deepcopy(optimizer.state_dict())
    previous_ema = deepcopy(ema_model.state_dict()) if ema_model is not None else None
    try:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        if ema_model is not None:
            ema_state = (
                state["ema_model"] if "ema_model" in state else model.state_dict()
            )
            ema_model.load_state_dict(ema_state)
    except BaseException:
        model.load_state_dict(previous_model)
        optimizer.load_state_dict(previous_optimizer)
        if ema_model is not None and previous_ema is not None:
            ema_model.load_state_dict(previous_ema)
        raise
    return step
