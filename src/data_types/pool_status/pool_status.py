from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PoolStatus:
    sample_count: int
    game_count: int
    ready: bool
