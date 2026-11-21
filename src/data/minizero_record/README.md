# MiniZero 文本棋谱解析

将一条 MiniZero Go 棋谱文本解析为 [`GameRecord`](game_record/game_record.py)，其中每手棋用 [`Move`](move/move.py) 表示。解析器只依赖 Python 标准库。 [实现代码](minizero_record.py)

## 用法

首版单机运行时，调用方从 MiniZero 的标准输出中提取一条完整的 `SelfPlay` 消息。以后从 PostgreSQL 读出一条棋谱文本，也可以直接交给同一个函数。 [采集命令](../../../third_party/README.md#自对弈)

```python
from src.data.minizero_record.minizero_record import parse_record

record_text = "SelfPlay true 2 2 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1]DLEN[0-1];B[81];W[81]) #"
record = parse_record(record_text)
```

`parse_record(text)` 每次接收一条记录，格式或字段无效时抛出 `RecordParseError`。标准输出的过滤和文件读取由调用方处理。

## 输入格式

- 自对弈使用 `SelfPlay true 140 140 -1 (;GM[go_9x9]…) #` 这样的消息。解析时用正文核对消息头里的手数、样本数和结果。 [消息格式](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/actor/actor_group.cpp#L24-L49)
- 导入已有棋谱时也可传入 `(;GM[go_9x9]SZ[9]KM[7]RE[-1];B[40]P[40:168,31:32]…)` 这样的正文。正文允许空白和换行，末尾可带服务器保存时附加的 `#`。走子和 `P` 都使用整数动作 ID，同一盘棋只有一条走子序列。 [棋谱序列化](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/environment/base/base_env.h#L207-L233) · [服务器保存格式](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/zero/zero_server.cpp#L263-L268)

## 复盘接口

| 字段 | 含义 |
| --- | --- |
| `board_size`、`komi`、`model_name` | 对应 `SZ`、`KM`、`EV`。`model_name` 可以为空。 |
| `black_result` | `RE` 的数值，黑胜 `+1`、白胜 `-1`、和棋 `0`。 |
| `moves` | 按走子顺序保存，每手含 `player` 和 `action_id`。黑为 `1`，白为 `2`。棋盘点从 `0` 编号，`pass` 是 `board_size ** 2`。 |
| `Move.policy_weights` | `P` 中按原顺序保存的 `(action_id, weight)` 对。项目当前 MCTS 配置下，权重是访问次数。缺失 `P` 或 `P[]` 时得到空元组。 |
| `Move.search_value`、`Move.reward` | 分别读取 `V` 和 `R`，字段缺失时为 `None`。 |
| `sample_range` | `DLEN` 从零开始且两端包含。解析后的 Python `range` 可直接用于判断 `move_index in record.sample_range`。缺省时覆盖所有已读走子。 |
| `is_complete` | `SelfPlay true` 或带 `#` 的裸棋谱为 `True`，`SelfPlay false` 为 `False`，无结束标记的裸棋谱为 `None`。 |

复盘从 `moves` 的第一手开始，到 `sample_range` 内的下标才对应训练位置。生成终局标签前需要确认记录完整，因为中间记录也会写入 `RE`。完整记录可以来自双方停一手或认输，规则终局由环境判断。解析阶段保留原始权重，合法性检查和策略归一化在复盘时处理。 [结果写入](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/actor/base_actor.cpp#L42-L64) · [策略权重读取](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/environment/base/base_env.h#L244-L283) · [环境检查](../check_minizero/README.md)

## 测试

```powershell
python -m unittest -v src.data.minizero_record.minizero_record_test
```

测试使用合成文本覆盖两种输入、停一手、`DLEN`、中间记录、转义和无效字段。 [测试代码](minizero_record_test.py)
