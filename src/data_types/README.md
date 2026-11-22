# 数据类

每个数据类有自己的目录，包含定义和字段说明。项目代码从对应的类型模块导入，已有调用方也可以通过旧路径导入同一个类。

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
| [`LearnerConfig`](learner_config/README.md) | 模型更新和经验池的配置。 |
| [`PoolStatus`](pool_status/README.md) | 经验池的当前规模和开训状态。 |
| [`StepMetrics`](step_metrics/README.md) | 一次模型更新的指标。 |
