# 棋盘旋转

`random_rotate_samples(samples, rng)` 接收一个 `TrainingSamples` 批次和 NumPy 随机数生成器。`Learner.step()` 在经验池完成采样后调用它，再将增强后的批次转为训练张量。 [实现代码](rotation.py) · [训练步](../README.md#训练步与检查点)

每条样本独立等概率选择逆时针旋转 `0°`、`90°`、`180°` 或 `270°`。旋转作用于所有特征通道与策略的棋盘落点。例如逆时针 `90°` 将数组中的 `(r, c)` 移到 `(n - 1 - c, r)`，对应的策略概率也移到同一落点。`pass` 概率和终局价值沿用原值。

函数返回独立的数组，保留批大小、`float32` 类型及原有行顺序。显式设置 `LearnerConfig.seed` 可复现取样和旋转序列。 [样本格式](../../../data_types/training_samples/README.md) · [随机种子](../../../data_types/learner_config/README.md)

## 测试

从仓库根目录运行。测试覆盖四种方向、所有特征通道与策略标签的对齐、`pass` 和价值、可复现性及空批次。 [测试代码](rotation_test.py)

```sh
python -m unittest -v src.core.learner.rotation.rotation_test
```
