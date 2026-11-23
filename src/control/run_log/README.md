# 运行日志

`RunLog(runs_dir, on_event=None)` 的 `open()` 创建本次运行的目录和日志。`write_metadata()` 保存 `run.json`，`event()` 将事件追加到 `events.jsonl`，写入后调用可选的终端回调。事件包含运行 ID 和时间，控制器提供轮次、步数、来源 UUID 和阶段所需的字段。

`log_step()` 在 `StepMetrics.step` 上记录损失、批大小和抽样时的池内样本数。`add_scalars()` 记录采集轮上的棋局统计、池规模、更新步数和耗时。`flush()` 在每轮结束时写出待处理的 TensorBoard 指标，`close()` 负责关闭输出。文件和 TensorBoard 错误由控制器报告。[查看 TensorBoard](../../../README.md#tensorboard)
