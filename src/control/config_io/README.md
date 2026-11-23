# 训练配置

用 `load_training_config(path)` 读入 TOML，返回 `TrainingConfig`。`[go].board_size` 同时设置模型和经验池的棋盘尺寸，两处显式提供的值须一致。 [实现代码](config_io.py) · [数据类](../../data_types/training_config/README.md) · [训练配置对照](../../../docs/training.md#选择训练配置)

`[run].pool_id`、模型的必填结构参数、优化器的两个参数，以及每轮棋谱数、每轮更新步数和最大轮数需要显式提供。其余参数使用数据类默认值。配置错误报告对应的表和字段，执行时依赖的文件由各阶段打开。

`[model]`、`[learner]`、`[optimizer]` 和 `[schedule]` 分别配置本项目的模型、参数更新、AdamW 和轮次。MiniZero 先加载 `[selfplay].config_path` 指定的基础配置，再接收 `training_overrides(config)` 从 `[go]`、`[selfplay]` 和 `[alphazero]` 生成的覆盖项。采集模块设置本轮的 `nn_file_name`。 [围棋规则映射](../../data_types/go_rules_config/README.md) · [采集参数映射](../../data_types/selfplay_config/README.md) · [搜索参数映射](../../data_types/alphazero_config/README.md#minizero-参数映射) · [采集模块](../selfplay_collector/README.md)

`save_training_config(config, path)` 使用 `tomli-w` 写入补齐默认值的 TOML，省略可选空值。TOML 中的相对路径以仓库根目录为基准。控制器保存的快照指向本次运行使用的 MiniZero 基础配置副本。 [控制器](../training_controller/README.md)

测试覆盖配置快照的往返读取和字段校验。 [测试代码](config_io_test.py)
