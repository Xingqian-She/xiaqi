# SelfPlayConfig

`engine_path` 和 `config_path` 指向 MiniZero Go 引擎和基础配置。`num_threads`、`num_parallel_games`、`seed`、`auto_seed` 和 `quiet` 控制采集进程。全新运行在预训练步数为零时可用 `initial_model` 指定首轮采集权重。[采集模块](../../control/selfplay_collector/README.md)
