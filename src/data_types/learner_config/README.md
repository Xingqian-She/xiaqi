# LearnerConfig

`LearnerConfig` 配置模型更新、经验池和训练批次。模型结构和优化器参数由传给 `Learner` 的工厂提供。 [类型定义](learner_config.py) · [调用示例](../../core/learner/README.md#调用)

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `board_size` | `9` | 棋盘边长，须与模型和样本一致。 |
| `device` | `"cpu"` | 放置模型和训练批次的设备，例如 `"cuda"`。 |
| `batch_size` | `256` | 每次更新抽取的样本数。 |
| `mini_batch_size` | `None` | 每次前向和反向传播的样本数。省略时使用整个 `batch_size`。 |
| `min_train_samples` | `10_000` | 开始更新所需的最少样本数。 |
| `target_samples` | `500_000` | 按整盘淘汰时的样本保留目标。 |
| `seed` | `None` | NumPy 随机种子。显式设值可复现抽样和旋转序列。 |
| `value_loss_weight` | `1.0` | 总损失中价值均方误差的权重，须有限且非负。 |
| `compile_model` | `False` | 启用后用 `torch.compile(model)` 运行训练前向。 |
| `ema_enabled` | `False` | 每次优化器更新后维护一份供推理使用的指数移动平均权重。 |
| `ema_tau` | `0.05` | 每次 EMA 更新时当前训练参数的权重，取值为 `(0, 1]`。 |

配置要求 `1 <= batch_size <= min_train_samples <= target_samples`。显式配置的 `mini_batch_size` 须为正整数，并能整除 `batch_size`。 [梯度累积](../../core/learner/gradient_accumulation/README.md) · [EMA 权重](../../core/learner/ema/README.md)
