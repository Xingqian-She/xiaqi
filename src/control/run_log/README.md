# 运行日志

`RunLog(directory, on_event=None)` 在选定实验目录中追加 `events.jsonl`，写入后调用可选的终端回调。每次启动产生新的 `run_id`。`run_started` 记录实际使用的 TOML、MiniZero 基础配置、初始权重和恢复点。后续事件记录轮次、步数、来源 UUID 和阶段所需的字段。 [实现代码](run_log.py) · [控制器](../training_controller/README.md) · [运行产物](../../../docs/training.md#运行产物与进度)

## TensorBoard 指标

`log_step()` 用累计训练步数写入 `tensorboard/train/`。`log_round()` 用实验中已完成的采集轮数写入 `tensorboard/rounds/`。恢复时，从检查点步数和最后一条完整轮次事件之后继续记录。`flush()` 在每轮结束时写出待处理的指标，`close()` 关闭输出。 [启动 TensorBoard](../../../docs/training.md#tensorboard)

| 横轴 | 指标 | 含义 |
| --- | --- | --- |
| 训练步 | `train/policy_loss`、`train/value_loss`、`train/total_loss` | 策略损失、价值损失和总损失。 |
| 训练步 | `train/batch_size`、`train/replay_samples` | 本步的批大小及抽样时池内的有效位置数。 |
| 采集轮 | `selfplay/games_committed`、`selfplay/games_scored`、`selfplay/games_with_samples` | 本批已提交、通过复盘、复盘后有样本的盘数。 |
| 采集轮 | `selfplay/black_wins`、`selfplay/white_wins`、`selfplay/draws` | 本批通过复盘的黑胜、白胜和和棋盘数。 |
| 采集轮 | `selfplay/black_win_rate`、`selfplay/white_win_rate`、`selfplay/draw_rate` | 三类结果各自占 `games_scored` 的比例。 |
| 采集轮 | `replay/sample_count`、`replay/game_count` | 增量复盘与淘汰后的有效位置数和活跃棋局数。 |
| 采集轮 | `round/train_steps`、`round/duration_seconds` | 本轮更新步数和耗时（秒）。 |

胜率按本轮固定的采集权重统计，分母为本轮 `source_run_id` 下通过复盘的完整棋局数，包含合法认输局和零样本局。分母为零时仅写盘数。运行汇总先合计各轮盘数，再计算比例。 [单步指标字段](../../data_types/step_metrics/README.md) · [统计实现](../training_controller/training_controller.py) · [控制器测试](../training_controller/training_controller_test.py)
