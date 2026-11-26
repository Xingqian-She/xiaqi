# SelfPlayConfig

`engine_path` 以仓库根目录为基准。基础配置 `config_path` 和外部首轮权重 `initial_model` 以实验目录为基准，默认基础配置为 `minizero_base.cfg`。`num_threads`、`num_parallel_games` 分别覆盖 `zero_num_threads`、`zero_num_parallel_games`。`seed`、`auto_seed` 和 `quiet` 分别覆盖 `program_seed`、`program_auto_seed` 和 `program_quiet`。 [类型定义](selfplay_config.py) · [采集模块](../../control/selfplay_collector/README.md)

全新运行在预训练步数为零时可用 `initial_model` 指定首轮采集的 TorchScript 权重。此时 Learner 按 `[model]` 初始化自身权重。默认使用新建模型导出首轮权重，续训从恢复的模型导出，设置了预训练步数时则从预训练后的模型导出。 [训练控制器](../../control/training_controller/README.md)
