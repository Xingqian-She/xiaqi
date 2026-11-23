from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """MiniZero 引擎路径、并发和初始采集权重。"""

    engine_path: Path = Path("build/minizero/go/minizero_go")
    config_path: Path = Path("configs/minizero/go_9x9_az.cfg")
    num_threads: int = 4
    num_parallel_games: int = 32
    seed: int = 0
    auto_seed: bool = True
    quiet: bool = False
    initial_model: Path | None = None

    def __post_init__(self) -> None:
        for name in ("num_threads", "num_parallel_games"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
