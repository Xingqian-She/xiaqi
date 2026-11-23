# 运行日志

`RunLog(runs_dir, on_event=None)` 的 `open()` 创建本次运行的目录和日志。`write_metadata()` 保存 `run.json`，`event()` 将事件追加到 `events.jsonl`，写入后调用可选的终端回调。事件包含运行 ID 和时间，控制器提供轮次、步数、来源 UUID 和阶段所需的字段。 [实现代码](run_log.py) · [控制器](../training_controller/README.md) · [运行产物](../../../docs/training.md#运行产物与进度)

## TensorBoard 指标

`log_step()` 以 `StepMetrics.step` 为横轴，控制器以采集轮号为横轴调用 `add_scalars()` 写入轮次指标。`flush()` 在每轮结束时写出待处理的指标，`close()` 关闭输出。文件和 TensorBoard 错误由控制器报告。 [启动 TensorBoard](../../../docs/training.md#tensorboard)

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
