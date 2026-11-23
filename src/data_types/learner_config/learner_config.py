import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    board_size: int = 9
    device: str | torch.device = "cpu"
    batch_size: int = 256
    min_train_samples: int = 10_000
    target_samples: int = 500_000
    seed: int | None = None
    value_loss_weight: float = 1.0
    compile_model: bool = False
    mini_batch_size: int | None = None
    ema_enabled: bool = False
    ema_tau: float = 0.05

    def __post_init__(self) -> None:
        for name, size in (
            ("board_size", self.board_size),
            ("batch_size", self.batch_size),
            ("min_train_samples", self.min_train_samples),
            ("target_samples", self.target_samples),
        ):
            if not isinstance(size, Integral) or isinstance(size, bool) or size < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.batch_size <= self.min_train_samples <= self.target_samples:
            raise ValueError(
                "Expected batch_size <= min_train_samples <= target_samples."
            )
        if not math.isfinite(self.value_loss_weight) or self.value_loss_weight < 0:
            raise ValueError("value_loss_weight must be finite and nonnegative.")
        mini_batch_size = self.mini_batch_size
        if mini_batch_size is not None:
            if (
                not isinstance(mini_batch_size, Integral)
                or isinstance(mini_batch_size, bool)
                or mini_batch_size < 1
            ):
                raise ValueError("mini_batch_size must be a positive integer.")
            if mini_batch_size > self.batch_size or int(self.batch_size) % int(
                mini_batch_size
            ):
                raise ValueError(
                    "mini_batch_size must divide batch_size and not exceed it."
                )
        if type(self.ema_enabled) is not bool:
            raise ValueError("ema_enabled must be a boolean.")
        if (
            not isinstance(self.ema_tau, Real)
            or isinstance(self.ema_tau, bool)
            or not 0 < self.ema_tau <= 1
        ):
            raise ValueError("ema_tau must be in (0, 1].")
