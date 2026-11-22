# TrainingSamples

`TrainingSamples` 表示一盘棋的复盘结果或用于训练的一个批次。三个 NumPy 数组均为 `float32`，同一行对应同一个训练位置。`N` 是样本数，`n` 是棋盘边长。 [类型定义](training_samples.py) · [复盘接口](../../data/minizero_replay/README.md#输出)

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `features` | `(N, 18, n, n)` | 落子前的局面特征，包含棋子历史和当前执子方。 |
| `policy` | `(N, n² + 1)` | 策略目标。棋盘落点按行展平，`pass` 位于下标 `n²`。 |
| `value` | `(N, 1)` | 当前执子方视角的终局标签，胜、和、负分别取 `1`、`0`、`-1`。 |

经验池直接保存入池数组的引用。调用方在入池后应将这些数组视为只读。 [入池校验](../../core/learner/replay_buffer/README.md#入池与淘汰)
