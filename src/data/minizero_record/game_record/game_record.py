from dataclasses import dataclass

from ..move.move import Move


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One parsed Go record and the move indices selected for training."""

    board_size: int
    komi: float
    black_result: int
    moves: tuple[Move, ...]
    sample_range: range
    model_name: str | None
    is_complete: bool | None
