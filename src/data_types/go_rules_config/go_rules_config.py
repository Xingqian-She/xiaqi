import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class GoRulesConfig:
    """自对弈和复盘共用的围棋规则。"""

    board_size: int = 9
    komi: float = 7.0
    ko_rule: Literal["positional", "situational"] = "positional"

    def __post_init__(self) -> None:
        if type(self.board_size) is not int or not 1 <= self.board_size <= 19:
            raise ValueError("board_size must be an integer between 1 and 19.")
        if not math.isfinite(self.komi):
            raise ValueError("komi must be finite.")
        if self.ko_rule not in ("positional", "situational"):
            raise ValueError("ko_rule must be positional or situational.")
