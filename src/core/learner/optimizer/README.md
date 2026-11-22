# 优化器参数分组

`prepare_optimizer(model, make_optimizer)` 将 `model.parameters()` 传给工厂，随后就地拆分返回的优化器参数组。每个含 `weight_decay` 的组按保留衰减、零衰减的顺序拆分。`cls_token`、已初始化的标量和一维参数进入零衰减组，覆盖偏置、归一化参数和 LayerScale 向量。子组继承原组的其他超参数，参数和 `param_names` 保留各自在组内的顺序。没有 `weight_decay` 的组原样使用。 [实现代码](optimizer.py)

函数返回原优化器及其状态转换函数。旧检查点沿用工厂的原始组数和参数顺序。恢复时，转换函数根据初始化时记录的分组位置同步拆分保存的参数 ID，使优化器将动量等状态加载到对应的参数。续训需使用匹配的模型和工厂分组。 [检查点与续训](../checkpoint/README.md)

## CPU 示例

从仓库根目录运行。这个 AdamW 优化器为线性层的权重保留 `0.02` 衰减，为偏置使用零衰减。

```python
from functools import partial

from torch import nn
from torch.optim import AdamW

from src.core.learner.optimizer.optimizer import prepare_optimizer

model = nn.Linear(3, 2)
optimizer, _ = prepare_optimizer(
    model, partial(AdamW, lr=1e-3, weight_decay=0.02)
)
names = {param: name for name, param in model.named_parameters()}
for group in optimizer.param_groups:
    print([names[param] for param in group["params"]], group["weight_decay"])
```

## 测试

```sh
python -m unittest -v src.core.learner.optimizer.optimizer_test
```
