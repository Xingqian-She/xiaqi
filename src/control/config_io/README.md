# 训练配置

用 `load_training_config(path)` 读入 TOML，返回 `TrainingConfig`。`[go].board_size` 同时设置模型和经验池的棋盘尺寸，两处显式提供的值须一致。[数据类](../../data_types/training_config/README.md) · [联调配置](../../../configs/training/go_9x9_smoke.toml)

`[run].pool_id`、模型的必填结构参数、优化器的两个参数，以及每轮棋谱数、每轮更新步数和最大轮数需要显式提供。其余参数使用数据类默认值。配置错误报告对应的表和字段，执行时依赖的文件由各阶段打开。

`save_training_config(config, path)` 使用 `tomli-w` 写入补齐默认值的 TOML，省略可选空值。TOML 中的相对路径以仓库根目录为基准。控制器保存的快照指向本次运行使用的 MiniZero 基础配置副本。[控制器](../training_controller/README.md)
