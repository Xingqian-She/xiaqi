import math
from dataclasses import dataclass
from numbers import Integral

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
