from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    policy_loss: float
    value_loss: float
    total_loss: float
    batch_size: int
    replay_samples: int
