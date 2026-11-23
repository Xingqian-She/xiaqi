import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AlphaZeroConfig:
    """MiniZero AlphaZero 自对弈的搜索、探索和认输选项。"""

    num_simulations: int = 200
    puct_base: float = 19652.0
    puct_init: float = 1.25
    reward_discount: float = 1.0
    value_rescale: bool = False
    action_selection: Literal["max_count", "softmax_count"] = "softmax_count"
    temperature: float = 1.0
    use_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.11
    dirichlet_epsilon: float = 0.25
    random_rotation_features: bool = True
    resign_threshold: float = -0.9
    disable_resign_ratio: float = 0.1

    def __post_init__(self) -> None:
        if type(self.num_simulations) is not int or self.num_simulations < 1:
            raise ValueError("num_simulations must be a positive integer.")
        for name in ("puct_base", "temperature", "dirichlet_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("dirichlet_epsilon", "disable_resign_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        for name in ("puct_init", "reward_discount", "resign_threshold"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.action_selection not in ("max_count", "softmax_count"):
            raise ValueError("action_selection must be max_count or softmax_count.")
