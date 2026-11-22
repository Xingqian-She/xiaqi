# PoolStatus

`add_samples()` 和 `pool_status()` 返回 `PoolStatus`，用于读取经验池的当前规模和开训状态。 [类型定义](pool_status.py) · [经验池](../../core/learner/replay_buffer/README.md)

| 字段 | 含义 |
| --- | --- |
| `sample_count` | 池中保留的有效训练位置总数。 |
| `game_count` | 池中保留的整盘数。 |
| `ready` | `sample_count >= min_train_samples` 时为 `True`。 |
