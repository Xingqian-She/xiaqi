# 训练控制器

`TrainingController(config, connection, on_event=None).run(resume=None, config_path=None)` 使用调用方管理的 PostgreSQL autocommit 连接，串行执行回填、采集、经验池刷新和模型更新。返回本次运行目录。命令行入口负责读取配置、建立连接并报告阶段进度。[命令和续训](../../../README.md#自对弈训练) · [配置](../config_io/README.md)

历史棋局从新到旧回填，增量棋局按提交顺序处理到本轮的最后编号。通过规则和复盘校验且产出样本的棋局加入经验池。训练盘数门槛只累计本控制器采集的有样本棋局。胜率统计使用本轮通过复盘的完整棋局，包含合法认输和零样本棋局。[训练调度](../../data_types/training_schedule/README.md) · [回填和增量读取](../../../PostgreSQL.md#内存窗口与采样)

`--resume` 从检查点恢复模型、优化器和训练步数，再回填经验池。正数预训练步数要求历史池就绪。每个完成的训练阶段先保存检查点，再导出采集模型。采集权重的路径、哈希和来源 UUID 可由运行事件追溯到数据库。[日志](../run_log/README.md)

解析、规则或复盘校验失败的棋局记录 ID、来源和原因后跳过。数据库、复盘环境或运行产物写入失败时结束运行，并在日志可用时记录 `run_failed`。中断记录 `run_interrupted`，留下的棋谱和检查点可用于下次运行。
