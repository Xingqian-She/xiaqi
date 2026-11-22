"""从内存经验池抽样并更新围棋策略与价值网络。"""

from collections.abc import Callable, Iterable
from copy import deepcopy
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from ...data_types.learner_config.learner_config import LearnerConfig
from ...data_types.pool_status.pool_status import PoolStatus
from ...data_types.step_metrics.step_metrics import StepMetrics
from ...data_types.training_samples.training_samples import TrainingSamples
from .replay_buffer.replay_buffer import ReplayBuffer
from .rotation.rotation import random_rotate_samples


class Learner:
    """持有训练模型、优化器和经验池，每次调用 step 更新一次参数。"""

    def __init__(
        self,
        make_model: Callable[[], nn.Module],
        make_optimizer: Callable[[Iterable[nn.Parameter]], Optimizer],
        config: LearnerConfig,
    ) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._replay_buffer = ReplayBuffer(
            config.board_size, config.min_train_samples, config.target_samples
        )
        self._rng = np.random.default_rng(config.seed)
        self._step = 0

        self._model = make_model().to(self._device)
        self._optimizer = make_optimizer(self._model.parameters())
        self._train_model = (
            torch.compile(self._model) if config.compile_model else self._model
        )

    def add_samples(
        self,
        samples: TrainingSamples,
        *,
        position: Literal["oldest", "newest"] = "newest",
    ) -> PoolStatus:
        return self._replay_buffer.add_samples(samples, position=position)

    def pool_status(self) -> PoolStatus:
        return self._replay_buffer.pool_status()

    def step(self) -> StepMetrics | None:
        status = self.pool_status()
        if not status.ready:
            return None

        batch_size = int(self._config.batch_size)
        batch = self._replay_buffer.sample_batch(batch_size, self._rng)
        batch = random_rotate_samples(batch, self._rng)
        board = torch.from_numpy(batch.features).to(self._device)
        policy = torch.from_numpy(batch.policy).to(self._device)
        value = torch.from_numpy(batch.value).to(self._device)

        self._train_model.train()
        self._optimizer.zero_grad(set_to_none=True)
        logits, value_pred = self._train_model(board)
        if logits.shape != policy.shape:
            raise ValueError(
                f"Expected policy logits shape {tuple(policy.shape)}, got {tuple(logits.shape)}."
            )
        if value_pred.shape != value.shape:
            raise ValueError(
                f"Expected value prediction shape {tuple(value.shape)}, "
                f"got {tuple(value_pred.shape)}."
            )

        policy_loss = -(policy * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
        value_loss = (value_pred - value).square().mean()
        total_loss = policy_loss + self._config.value_loss_weight * value_loss
        total_loss.backward()
        self._optimizer.step()
        self._step += 1
        return StepMetrics(
            step=self._step,
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            total_loss=total_loss.item(),
            batch_size=batch_size,
            replay_samples=status.sample_count,
        )

    def save_checkpoint(self, path: str | PathLike[str]) -> None:
        """保存训练状态，写入完成后替换目标文件。"""
        destination = Path(path)
        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "step": self._step,
        }
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

    def load_checkpoint(self, path: str | PathLike[str]) -> None:
        """就地恢复模型与优化器，保留训练前向对同一模型的引用。"""
        state = torch.load(path, map_location="cpu", weights_only=True)
        model_state = state["model"]
        optimizer_state = state["optimizer"]
        step = state["step"]
        if type(step) is not int or step < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer.")

        previous_model = deepcopy(self._model.state_dict())
        previous_optimizer = deepcopy(self._optimizer.state_dict())
        try:
            self._model.load_state_dict(model_state)
            self._optimizer.load_state_dict(optimizer_state)
        except Exception:
            self._model.load_state_dict(previous_model)
            self._optimizer.load_state_dict(previous_optimizer)
            raise
        self._step = step

    def get_inference_model(self) -> nn.Module:
        """返回调用方在训练步之间用于只读访问的原始模型。"""
        return self._model
