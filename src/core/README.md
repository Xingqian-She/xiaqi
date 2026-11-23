# Core

`src/core/` 存放项目自身的模型和学习组件。它使用 PyTorch 和 NumPy 进行计算，以棋盘张量和项目定义的 `TrainingSamples` 作为输入。采集引擎或棋谱存储方式变化时，只要上游仍提供相同格式的样本，模型和 Learner 就可以继续复用。 [训练样本格式](../data_types/training_samples/README.md)

| 模块 | 职责 |
| --- | --- |
| [围棋模型](model/README.md) | 用卷积 Stem 和 Transformer Encoder 处理棋盘特征，输出策略 logits 和当前执子方的价值。 |
| [Learner](learner/README.md) | 管理内存经验池和样本增强，执行参数更新，保存与恢复训练检查点。 |

数据模块负责 MiniZero 棋谱的采集、复盘和 PostgreSQL 存取，训练控制模块负责安排采集与更新的轮次。MiniZero 模型适配模块负责将 Core 的模型导出为引擎可加载的 TorchScript。 [数据模块](../data/README.md) · [训练控制模块](../control/README.md) · [MiniZero 模型适配](../integrations/minizero/README.md)
