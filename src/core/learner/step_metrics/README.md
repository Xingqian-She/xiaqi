# StepMetrics

`StepMetrics` 是 `Learner.step()` 完成一次参数更新后返回的指标。三个损失字段均为 Python 浮点数。 [类型定义](step_metrics.py) · [训练步](../README.md#训练步与检查点)

| 字段 | 含义 |
| --- | --- |
| `step` | 已完成的更新步数，从 `1` 开始。 |
| `policy_loss` | 策略软标签交叉熵，动作包含 `pass`。 |
| `value_loss` | 加权前的价值均方误差。 |
| `total_loss` | `policy_loss + value_loss_weight * value_loss`。 |
| `batch_size` | 本次更新使用的样本数。 |
| `replay_samples` | 抽样时经验池中的有效训练位置数。 |
