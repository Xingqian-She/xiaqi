from dataclasses import dataclass

from ..alphazero_config.alphazero_config import AlphaZeroConfig
from ..go_rules_config.go_rules_config import GoRulesConfig
from ..learner_config.learner_config import LearnerConfig
from ..optimizer_config.optimizer_config import OptimizerConfig
from ..selfplay_config.selfplay_config import SelfPlayConfig
from ..training_schedule.training_schedule import TrainingSchedule
from ..transformer_config.transformer_config import TransformerConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """一次训练运行所需的配置。"""

    go: GoRulesConfig
    model: TransformerConfig
    learner: LearnerConfig
    optimizer: OptimizerConfig
    selfplay: SelfPlayConfig
    alphazero: AlphaZeroConfig
    schedule: TrainingSchedule
