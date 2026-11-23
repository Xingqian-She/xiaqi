# 训练检查点

`Learner.save_checkpoint(path)` 保存原始模型状态、优化器状态和已完成步数，对应 `model`、`optimizer` 和 `step` 三个字段。调用方准备目标目录并在训练步之间调用。模块将状态写到同目录的临时文件，写入成功后替换目标，失败时清理临时文件。 [实现代码](checkpoint.py)

`Learner.load_checkpoint(path)` 在 CPU 上读取检查点，再就地恢复模型和优化器。优化器状态转换由初始化时的参数分组提供，恢复失败时回滚模型和优化器。成功后，`Learner` 更新步数。编译训练和普通训练共用同一格式。续训时用匹配的模型与优化器工厂创建 `Learner`，加载检查点后回填经验池。 [优化器分组](../optimizer/README.md) · [Learner 调用](../README.md)

## CPU 示例

从仓库根目录运行。示例用合成样本完成第 1 步，保存后创建新的 `Learner`，加载并回填样本以完成第 2 步。

```python
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from torch.optim import AdamW

from src.core.learner.learner import Learner
from src.core.model.transformer.transformer import Transformer
from src.data_types.learner_config.learner_config import LearnerConfig
from src.data_types.training_samples.training_samples import TrainingSamples

make_model = partial(
    Transformer,
    board_size=3,
    in_channels=18,
    model_dim=8,
    num_heads=2,
    ffn_hidden_dim=12,
    num_layers=1,
    value_hidden_dim=4,
)
make_optimizer = partial(AdamW, lr=1e-3, weight_decay=1e-4)
config = LearnerConfig(board_size=3, batch_size=1, min_train_samples=1, target_samples=1)
policy = np.zeros((1, 10), dtype=np.float32)
policy[0, -1] = 1.0
samples = TrainingSamples(
    features=np.zeros((1, 18, 3, 3), dtype=np.float32),
    policy=policy,
    value=np.ones((1, 1), dtype=np.float32),
)

learner = Learner(make_model, make_optimizer, config)
learner.add_samples(samples)
print(learner.step().step)  # 1

with TemporaryDirectory() as directory:
    path = Path(directory) / "learner.pt"
    learner.save_checkpoint(path)
    resumed = Learner(make_model, make_optimizer, config)
    resumed.load_checkpoint(path)

resumed.add_samples(samples)
print(resumed.step().step)  # 2
```

## 测试

测试覆盖训练状态恢复、旧检查点兼容和加载失败时的回滚。 [测试代码](checkpoint_test.py)

```sh
python -m unittest -v src.core.learner.checkpoint.checkpoint_test
```
