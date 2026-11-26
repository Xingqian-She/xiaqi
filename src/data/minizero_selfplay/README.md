# MiniZero 自对弈采集

从 MiniZero 的标准输出或已有输出文件中逐行取得完整对局消息。现场采集时，本模块还负责导出生效配置和管理 MiniZero 子进程。独立采集脚本和训练控制器通过共用采集模块登记批次并逐局入库。[实现代码](minizero_selfplay.py) · [共用采集模块](../../control/selfplay_collector/README.md)

## 读取已有输出

`iter_complete_records(lines)` 接收任意按行迭代的文本流。下面的示例从已保存的 MiniZero 输出 `record.txt` 中打印完整对局消息。

```python
from pathlib import Path

from src.data.minizero_selfplay.minizero_selfplay import iter_complete_records

with Path("record.txt").open(encoding="utf-8") as lines:
    for record in iter_complete_records(lines):
        print(record)
```

完整消息形如 `SelfPlay true 2 2 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1];B[81];W[81]) #`。函数要求行首为 `SelfPlay true `，行尾为 ` #`，允许 `#` 后有空白。普通日志、`SelfPlay false` 中间记录和缺少结束标记的行都会跳过。返回的字符串只移除行尾换行符。筛选只检查消息首尾，正文的字段校验由 `parse_record()` 负责。 [棋谱格式与解析](../minizero_record/README.md)

## 现场自对弈

`read_effective_config(command)` 和 `selfplay_stdout(command)` 接收同一组基础参数。`command` 包含 MiniZero 可执行文件及 `-conf_file`、`-conf_str` 等配置参数。

| 接口 | 行为 |
| --- | --- |
| `read_effective_config(command)` | 追加 `-gen <临时文件>`，返回 MiniZero 应用覆盖参数后写出的配置全文。引擎返回非零退出码时抛出 `RuntimeError`。 |
| `selfplay_stdout(command)` | 追加 `-mode sp` 启动引擎，向标准输入发送 `start`，通过 `with` 将标准输出流交给调用方。该输出流可直接传给 `iter_complete_records()`。 |

退出 `with` 时，如果引擎仍在运行，函数尝试发送 `quit` 并排空剩余输出，等待超过 10 秒则终止进程。收尾期间读到的剩余输出会被丢弃。调用方正常离开 `with` 时，引擎的非零退出码会报告为 `RuntimeError`。

在所选实验的数据库 Shell 中运行 `python -m scripts.collect_selfplay`。`--games N` 现场生成并写入 N 条完整对局消息，先筛选再计数。`--record-file PATH` 读取已有输出文件，调用方需传入原采集批次的元数据。 [构建引擎与准备权重](../../../third_party/README.md#构建) · [完整采集和导入命令](../../../PostgreSQL.md#采集与导入)
