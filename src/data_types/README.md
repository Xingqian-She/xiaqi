# 数据类

每个数据类有自己的目录，包含字段说明和类型定义。项目代码从对应的类型模块导入。

```python
from src.data_types.source_run.source_run import SourceRun
```

| 类型 | 用途 |
| --- | --- |
| [`SourceRun`](source_run/README.md) | 自对弈采集批次的来源配置。 |
| [`StoredGame`](stored_game/README.md) | 从 PostgreSQL 读出的一盘棋谱及其来源。 |
| [`GameRecord`](game_record/README.md) | 解析后的对局与训练采样范围。 |
| [`Move`](move/README.md) | 一手棋的动作与搜索数据。 |
| [`TrainingSamples`](training_samples/README.md) | 复盘或采样得到的训练数组。 |
| [`TransformerConfig`](transformer_config/README.md) | 围棋 Transformer 的棋盘尺寸和网络结构配置。 |
| [`LearnerConfig`](learner_config/README.md) | 模型更新和经验池的配置。 |
| [`PoolStatus`](pool_status/README.md) | 经验池的当前规模和开训状态。 |
| [`StepMetrics`](step_metrics/README.md) | 一次模型更新的指标。 |
| [`TrainingConfig`](training_config/README.md) | 训练控制器的总配置和运行目录。 |
| [`TrainingSchedule`](training_schedule/README.md) | 采集轮次、预训练和常规更新门槛。 |
| [`GoRulesConfig`](go_rules_config/README.md) | 采集、复盘和训练共用的围棋规则。 |
| [`AlphaZeroConfig`](alphazero_config/README.md) | 搜索、探索和认输参数。 |
| [`SelfPlayConfig`](selfplay_config/README.md) | 引擎路径、并发和初始采集权重。 |
| [`OptimizerConfig`](optimizer_config/README.md) | AdamW 的学习率和权重衰减。 |
| [`SelfPlayResult`](selfplay_result/README.md) | 采集或导入确认提交的盘数和来源。 |
