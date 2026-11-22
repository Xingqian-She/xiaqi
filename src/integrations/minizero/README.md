# MiniZero TorchScript 导出

`export_minizero_torchscript(model, path)` 将围棋 Transformer 的当前权重保存为 MiniZero 可加载的 `.pt`。模型需要 18 个输入通道，棋盘边长最大为 19，并与采集配置的 `env_board_size` 一致。导出使用 CPU 上的 `float32` 推理副本。调用方准备好目标目录，临时文件写完后会原子替换目标文件。 [导出实现](torchscript_export.py)

导出的模型提供 MiniZero 读取的 12 个元数据方法。网络类型是 `alphazero`，`get_num_blocks()` 返回 Transformer 的 Encoder 层数。对 9×9 输入，`policy_logit` 和 `policy` 的形状均为 `[B, 82]`，最后一项为 `pass`，82 项一起做 softmax。`value` 的形状为 `[B, 1]`，黑方回合保留模型预测，白方回合将预测取反，以黑方视角交给 MiniZero 搜索。 [Transformer 接口](../../core/model/transformer/README.md#接口与形状)

## 导出小模型

在仓库根目录运行以下命令，生成一个随机初始化的小模型以检查导出和采集。

```bash
python - <<'PY'
from pathlib import Path

from src.core.model.transformer.transformer import Transformer
from src.data_types.transformer_config.transformer_config import TransformerConfig
from src.integrations.minizero.torchscript_export import export_minizero_torchscript

config = TransformerConfig(
    board_size=9,
    in_channels=18,
    model_dim=8,
    num_heads=2,
    ffn_hidden_dim=8,
    num_layers=1,
    value_hidden_dim=4,
)
path = Path("models/minizero/tiny_go_9x9.pt")
path.parent.mkdir(parents=True, exist_ok=True)
export_minizero_torchscript(Transformer.from_config(config), path)
PY
```

导出训练中的权重时，在两次训练步之间调用 `export_minizero_torchscript(learner.get_inference_model(), path)`。 [Learner 接口](../../core/learner/README.md#训练步与检查点)

## 用导出模型采集一盘

在按当前 Dockerfile 构建的 GPU `dev` 容器中，用 `bash scripts/build_minizero.sh` 构建引擎及其提交号文件。准备 PostgreSQL 表和 `XIAQI_DATABASE_URL` 后，在仓库根目录运行以下命令。 [数据库准备](../../../README.md#自对弈棋谱入库) · [MiniZero 构建](../../../third_party/README.md#构建)

```bash
python -m scripts.collect_selfplay --pool-id tiny-go-9x9 --games 1 \
  --config configs/minizero/go_9x9_az.cfg --model models/minizero/tiny_go_9x9.pt \
  --conf-str 'program_quiet=true:zero_num_threads=1:zero_num_parallel_games=1:actor_num_simulation=2:zero_disable_resign_ratio=1'
```

## 测试

CPU 单元测试核对重载后的元数据和前向结果、黑白回合的价值转换、原训练模型状态和保存失败时的文件完整性。 [测试代码](torchscript_export_test.py)

```bash
python -m unittest -v src.integrations.minizero.torchscript_export_test.TorchScriptExportTest
```

现场采集用例使用上面的 9×9 配置和真实 MiniZero 引擎。它在 120 秒内通过采集脚本入库一盘，检查完整棋谱和模型 SHA-256。测试在独立的 PostgreSQL schema 中建表，并在结束时清理该 schema。下面的命令复用已配置的开发数据库连接串，也可将 `XIAQI_TEST_DATABASE_URL` 指向允许创建 schema 的测试数据库。

```bash
XIAQI_TEST_MINIZERO_SELFPLAY=1 XIAQI_TEST_DATABASE_URL="$XIAQI_DATABASE_URL" \
  python -m unittest -v src.integrations.minizero.torchscript_export_test.MiniZeroSelfPlayTest
```
