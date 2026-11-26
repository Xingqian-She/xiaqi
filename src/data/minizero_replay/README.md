# MiniZero 围棋复盘与训练样本

`replay_record(record, *, ko_rule="positional")` 接收解析后的 `GameRecord`，用 `minizero_py` 从第一手复盘整局，为 `DLEN` 选中的每一手生成落子前的 AlphaZero 样本。 [实现代码](minizero_replay.py) · [棋谱解析](../minizero_record/README.md)

## 调用

在 `dev` 容器中运行，并将 Go 版 `minizero_py` 所在的 `/workspace/build/minizero/go` 加入 `PYTHONPATH`。棋盘尺寸和贴目取自棋谱，`ko_rule` 应与采集配置一致，接受 `positional` 和 `situational`。默认值沿用项目的 `positional`。MiniZero 配置在进程内共享，同一进程内应串行调用。 [构建 Python 绑定](../check_minizero/README.md#构建与运行) · [9×9 配置](../../../runs/go-9x9-az/minizero_base.cfg)

```python
from src.data.minizero_record.minizero_record import parse_record
from src.data.minizero_replay.minizero_replay import replay_record

text = (
    "SelfPlay true 4 4 -1 "
    "(;GM[go_9x9]SZ[9]KM[7]RE[-1]DLEN[0-3]"
    ";B[1]P[1:3,81:1];W[80]P[80:2,81:2]"
    ";B[81]P[4:1,81:2];W[81]P[81:1]) #"
)
samples = replay_record(parse_record(text))
print(samples.features.shape, samples.policy.shape, samples.value.shape)
# (4, 18, 9, 9) (4, 82) (4, 1)
```

## 输出

返回的 `TrainingSamples` 第 `i` 行对应棋谱的走子下标 `record.sample_range.start + i`。`features` 取该手落子前的特征，`policy` 将 `P` 按权重总和归一化，未列出的动作概率为 `0`。`value` 在黑方回合取 `RE`，白方回合取 `-RE`。 [样本字段与形状](../../data_types/training_samples/README.md)

## 校验

选中的训练位置缺失 `P` 或出现 `P[]` 时抛出 `ReplayError`。候选动作须合法且唯一，权重须非负且总和为有限正数。校验失败时也抛出 `ReplayError`，报错中的 `Move` 从第 1 手开始计数。`DLEN` 范围外的走子仍参与复盘，以保留局面历史并核对终局。若棋谱文本里的 `P` 本身格式无效，`parse_record` 会先报 `RecordParseError`。

复盘校验整局落子、执子顺序和终局结果。一条没有完整标记的棋谱须走到规则终局。带完整标记且未到规则终局的记录按认输计分，`SelfPlay false` 中间记录会报错。整局通过检查后才返回样本。 [环境接口](../check_minizero/README.md#复盘会用到的约定)

## 测试

构建 Go 版 Python 绑定后，从仓库根目录运行。

```powershell
docker compose run --rm --no-deps --pull never -T -e PYTHONPATH=/workspace/build/minizero/go dev `
  python -m unittest -v src.data.minizero_replay.minizero_replay_test
```

测试使用合成棋谱，在真实绑定上核对落子前特征、策略目标、黑白价值视角、`DLEN`、终局计分和无效 `P`。找不到 `minizero_py` 时跳过原生绑定用例。 [测试代码](minizero_replay_test.py)
