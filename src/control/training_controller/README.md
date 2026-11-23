# 训练控制器

`TrainingController(config, connection, on_event=None).run(resume=None, config_path=None)` 使用调用方管理的 PostgreSQL autocommit 连接，串行执行回填、采集、经验池刷新和模型更新。返回本次运行目录。命令行入口负责读取配置、建立连接并报告阶段进度。 [实现代码](training_controller.py) · [命令和续训](../../../docs/training.md) · [配置](../config_io/README.md)

历史棋局从新到旧回填，增量棋局按提交顺序处理到本轮的最后编号。通过规则和复盘校验且产出样本的棋局加入经验池。训练盘数门槛只累计本控制器采集的有样本棋局。胜率统计使用本轮通过复盘的完整棋局，包含合法认输和零样本棋局。[训练调度](../../data_types/training_schedule/README.md) · [回填和增量读取](../../../PostgreSQL.md#内存窗口与采样)

`--resume` 从检查点恢复模型、优化器和训练步数，再回填经验池。正数预训练步数要求历史池就绪。控制器在预训练完成、到达配置的保存轮次及正常结束时调用 `Learner.save_checkpoint()` 保存尚未落盘的更新。自对弈使用 `/dev/shm` 临时目录中的同一个 `model.pt`，在下一轮采集前发布更新后的推理模型。模型版本可用来源 UUID 和 SHA-256 从运行事件追溯到数据库。 [日志](../run_log/README.md)

解析、规则或复盘校验失败的棋局记录 ID、来源和原因后跳过。数据库、复盘环境或运行产物写入失败时结束运行，并在日志可用时记录 `run_failed`。Ctrl+C 在已开始的更新步完成或采集子进程收尾后保存一致的训练状态。控制器记录 `run_interrupted` 并抛出 `KeyboardInterrupt` 的子类 `TrainingInterrupted`，携带可用的检查点路径及保存错误。

测试覆盖多轮权重接续、按轮保存及中断续训、累计盘数门槛、胜率统计和故障收尾。 [测试代码](training_controller_test.py)
