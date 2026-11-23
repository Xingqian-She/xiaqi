"""从内存经验池抽样并更新围棋策略与价值网络。"""

from collections.abc import Callable, Iterable
from os import PathLike
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from ...data_types.learner_config.learner_config import LearnerConfig
from ...data_types.pool_status.pool_status import PoolStatus
from ...data_types.step_metrics.step_metrics import StepMetrics
from ...data_types.training_samples.training_samples import TrainingSamples
from .checkpoint import checkpoint
from .ema.ema import ExponentialMovingAverage
from .gradient_accumulation.gradient_accumulation import train_accumulated_batch
from .optimizer.optimizer import prepare_optimizer
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
        self._optimizer, self._normalize_optimizer_state = prepare_optimizer(
            self._model, make_optimizer
        )
        self._ema = (
            ExponentialMovingAverage(self._model, config.ema_tau)
            if config.ema_enabled
            else None
        )
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

    @property
    def step_count(self) -> int:
        """当前模型已完成的更新步数。"""
        return self._step

    def step(self) -> StepMetrics | None:
        status = self.pool_status()
        if not status.ready:
            return None

        batch_size = int(self._config.batch_size)
        batch = self._replay_buffer.sample_batch(batch_size, self._rng)
        batch = random_rotate_samples(batch, self._rng)
        policy_loss, value_loss, total_loss = train_accumulated_batch(
            self._train_model,
            self._optimizer,
            batch,
            device=self._device,
            mini_batch_size=int(self._config.mini_batch_size or batch_size),
            value_loss_weight=self._config.value_loss_weight,
        )
        if self._ema is not None:
            self._ema.update()
        self._step += 1
        return StepMetrics(
            step=self._step,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            batch_size=batch_size,
            replay_samples=status.sample_count,
        )

    def save_checkpoint(self, path: str | PathLike[str]) -> None:
        """保存训练状态，写入完成后替换目标文件。"""
        checkpoint.save_checkpoint(
            path,
            model=self._model,
            optimizer=self._optimizer,
            step=self._step,
            ema_model=self._ema.model if self._ema is not None else None,
        )

    def load_checkpoint(self, path: str | PathLike[str]) -> None:
        """就地恢复训练状态，保留训练前向对同一模型的引用。"""
        self._step = checkpoint.load_checkpoint(
            path,
            model=self._model,
            optimizer=self._optimizer,
            normalize_optimizer_state=self._normalize_optimizer_state,
            ema_model=self._ema.model if self._ema is not None else None,
        )

    def get_inference_model(self) -> nn.Module:
        """返回调用方在训练步之间用于只读访问的推理模型。"""
        return self._ema.model if self._ema is not None else self._model
