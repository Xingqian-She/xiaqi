import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """构造 AdamW 所用的学习率和权重衰减。"""

    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        for name in ("lr", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
