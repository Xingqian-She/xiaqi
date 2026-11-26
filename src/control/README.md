# 训练控制模块

训练入口读取配置并连接 PostgreSQL，控制器按轮串行执行自对弈、经验池刷新和模型更新。 [命令与续训](../../docs/training.md) · [Python 入口](../../scripts/train.py)

| 模块 | 职责 |
| --- | --- |
| [训练控制器](training_controller/README.md) | 启动回填、安排采集与训练、管理检查点和推理模型。 |
| [训练配置](config_io/README.md) | 加载和校验实验的训练 TOML，解析文件路径。 |
| [自对弈采集与导入](selfplay_collector/README.md) | 组装 MiniZero 参数、登记采集来源并逐盘提交完整棋谱。 |
| [运行日志](run_log/README.md) | 保存运行元数据、阶段事件和 TensorBoard 指标。 |
