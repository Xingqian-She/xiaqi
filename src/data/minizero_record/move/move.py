from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Move:
    """One recorded action with optional search data and reward."""

    player: int
    action_id: int
    policy_weights: tuple[tuple[int, float], ...] = ()
    search_value: float | None = None
    reward: float | None = None
