# 指数移动平均权重

设置 `[learner].ema_enabled = true` 后，Learner 用训练模型的初始权重创建独立的推理副本。副本与训练模型放在同一设备上，使用评估模式，参数无需梯度。每次优化器成功更新后调用一次 `ExponentialMovingAverage.update()`。 [实现代码](ema.py) · [Learner 配置](../../../data_types/learner_config/README.md)

浮点参数按 `EMA ← (1 − tau) × EMA + tau × 当前参数` 更新。`0 < tau <= 1`，默认值 `0.05` 对应约 13.5 次优化器更新的半衰期。缓冲区直接复制训练模型的当前值，包括计数器和运行统计量。

`get_inference_model()` 返回当前推理权重供只读访问。在训练步结束时保存检查点，会同时写入本次更新后的训练权重和 EMA 权重。旧检查点可以用恢复后的训练权重初始化 EMA。 [检查点](../checkpoint/README.md) · [MiniZero 权重导出](../../../integrations/minizero/README.md)

## 测试

从仓库根目录运行。用例覆盖连续两次参数平均、浮点和整型缓冲区同步，以及 `tau = 1`。 [测试代码](ema_test.py)

```sh
python -m unittest -v src.core.learner.ema.ema_test
```
