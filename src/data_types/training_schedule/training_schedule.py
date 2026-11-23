from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingSchedule:
    """自对弈轮次、模型更新和检查点的触发条件。"""

    games_per_round: int
    train_steps_per_round: int
    max_rounds: int
    warmup_train_steps: int = 0
    min_new_games_before_train: int = 0
    checkpoint_every_rounds: int = 1

    def __post_init__(self) -> None:
        for name, minimum in (
            ("games_per_round", 1),
            ("train_steps_per_round", 1),
            ("max_rounds", 1),
            ("warmup_train_steps", 0),
            ("min_new_games_before_train", 0),
            ("checkpoint_every_rounds", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
