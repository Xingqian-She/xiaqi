# 模型更新

`Learner` 持有 PyTorch 模型、优化器和 CPU 经验池。每次 `step()` 抽取一个批次，更新一次参数并返回 `StepMetrics`。入口负责复盘、安排训练步数和管理检查点的保存路径。 [实现代码](learner.py) · [训练控制器](../../control/training_controller/README.md) · [经验池](replay_buffer/README.md)

## 调用

下面从一盘 9×9 棋谱生成四条样本并执行一次更新。运行示例需在 `dev` 容器中准备 Go 版 `minizero_py`。模型工厂将模型的棋盘尺寸和输入通道与复盘数据对齐，优化器工厂设置学习率和权重衰减。 [复盘环境](../../data/minizero_replay/README.md#调用) · [Transformer](../model/transformer/README.md) · [优化器分组](optimizer/README.md)

```python
from functools import partial

from torch.optim import AdamW

from src.core.learner.learner import Learner
from src.core.model.transformer.transformer import Transformer
from src.data.minizero_record.minizero_record import parse_record
from src.data.minizero_replay.minizero_replay import replay_record
from src.data_types.learner_config.learner_config import LearnerConfig

make_model = partial(
    Transformer,
    board_size=9,
    in_channels=18,
    model_dim=32,
    num_heads=4,
    ffn_hidden_dim=48,
    num_layers=2,
    value_hidden_dim=16,
)
config = LearnerConfig(batch_size=2, min_train_samples=4, target_samples=12, seed=0)
learner = Learner(make_model, partial(AdamW, lr=1e-3, weight_decay=1e-4), config)

record = parse_record(
    "(;GM[go_9x9]SZ[9]KM[7]RE[-1]DLEN[0-3]"
    ";B[1]P[1:3,81:1];W[80]P[80:2,81:2];B[81]P[4:1,81:2];W[81]P[81:1])"
)
learner.add_samples(replay_record(record))
print(learner.step())
```

经验池达到 `min_train_samples` 后即可训练。`target_samples` 控制经验池的保留窗口。显式指定 `seed` 可复现抽样和棋盘旋转。 [配置项与默认值](../../data_types/learner_config/README.md)

## 训练步与检查点

- `add_samples(samples, position="newest")` 按盘加入已有 `TrainingSamples`，返回 `PoolStatus`。历史棋局按从新到旧的顺序回填时使用 `position="oldest"`。
- `pool_status()` 返回有效位置数、活跃棋局数和就绪状态。`step()` 在池未就绪时返回 `None`。就绪后对抽到的每条样本随机旋转棋盘特征和策略落点，再完成一次参数更新并返回 `StepMetrics`。 [棋盘旋转](rotation/README.md) · [指标字段](../../data_types/step_metrics/README.md)
- `save_checkpoint(path)` 将原始模型状态、优化器状态和步数写到指定路径，目标目录由入口准备。续训时用匹配的模型和优化器工厂创建 `Learner`，调用 `load_checkpoint(path)` 后从棋谱重建经验池。`step_count` 返回当前模型已完成的更新步数，包含检查点恢复的步数。编译与普通训练共用检查点格式。[检查点与续训示例](checkpoint/README.md)
- `get_inference_model()` 返回持有当前权重的原始模型引用。入口在训练步之间将它传给导出函数，由导出函数准备推理副本。 [MiniZero 推理导出](../../integrations/minizero/README.md)

## 测试

从仓库根目录运行。训练流程测试直接构造 `TrainingSamples`。优化器与检查点各有独立测试，整套测试可在有 NumPy 和 PyTorch 的 CPU 环境执行。 [Learner 测试代码](learner_test.py)

```sh
python -m unittest -v src.core.learner.learner_test \
  src.core.learner.optimizer.optimizer_test \
  src.core.learner.checkpoint.checkpoint_test \
  src.core.learner.replay_buffer.replay_buffer_test \
  src.core.learner.rotation.rotation_test
```
