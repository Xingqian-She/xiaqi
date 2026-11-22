# LearnerConfig

`LearnerConfig` 配置模型更新、经验池和训练批次。模型结构和优化器参数由传给 `Learner` 的工厂提供。 [类型定义](learner_config.py) · [调用示例](../README.md#调用)

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `board_size` | `9` | 棋盘边长，须与模型和样本一致。 |
| `device` | `"cpu"` | 放置模型和训练批次的设备，例如 `"cuda"`。 |
| `batch_size` | `256` | 每次更新抽取的样本数。 |
| `min_train_samples` | `10_000` | 开始更新所需的最少样本数。 |
| `target_samples` | `500_000` | 按整盘淘汰时的样本保留目标。 |
| `seed` | `None` | NumPy 随机种子。显式设值可复现抽样和旋转序列。 |
| `value_loss_weight` | `1.0` | 总损失中价值均方误差的权重，须有限且非负。 |
| `compile_model` | `False` | 启用后用 `torch.compile(model)` 运行训练前向。 |

配置要求 `1 <= batch_size <= min_train_samples <= target_samples`。小规模联调可以同时调低三个值。
