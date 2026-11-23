# 梯度累积

`Learner.step()` 抽取并旋转一个完整批次，把 CPU 上的 `TrainingSamples` 交给 `train_accumulated_batch()`。函数逐个把等长的 mini batch 搬到训练设备，计算策略交叉熵和价值均方误差，将用于反向传播的总损失除以 mini batch 数量。所有 mini batch 完成后，优化器更新一次。返回的三个损失均为完整批次的均值。 [实现代码](gradient_accumulation.py) · [样本形状](../../../data_types/training_samples/README.md)

例如 `batch_size = 1024`、`mini_batch_size = 512` 时，每步执行两次前向和反向传播。两次损失各按 `1/2` 缩放，显存中的训练激活按 512 条样本计算。mini batch 必须为正整数，至多等于批大小，且能整除批大小。省略 mini batch 时，整个 batch 一次完成。分批过程中出错会清除已累积的梯度，重新执行时从清空梯度开始。 [Learner 配置](../../../data_types/learner_config/README.md)

## 测试

从仓库根目录运行。用例将分批后的权重、优化器状态和损失与独立的整批更新比较，并验证第二个 mini batch 出错后的重试。 [测试代码](gradient_accumulation_test.py)

```sh
python -m unittest -v src.core.learner.gradient_accumulation.gradient_accumulation_test
```
