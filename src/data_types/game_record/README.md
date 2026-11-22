# GameRecord

`GameRecord` 是 `parse_record(text)` 的返回值，包含一条围棋记录的对局信息、走子顺序和训练采样范围。 [实现代码](game_record.py) · [解析入口](../../data/minizero_record/minizero_record.py)

| 字段 | 含义 |
| --- | --- |
| `board_size`、`komi`、`model_name` | 棋盘边长、贴目和模型文件名。棋谱可以省略模型名。 |
| `black_result` | 黑方视角的结果，`+1` 是黑胜，`-1` 是白胜，`0` 是和棋。 |
| `moves` | 按对局顺序保存的 [`Move`](../move/README.md) 元组。 |
| `sample_range` | 选作训练样本的走子下标，从 `0` 开始。例如 `range(1, 3)` 对应第二手和第三手。 |
| `is_complete` | `True` 表示记录标记为完整，`False` 表示中间记录，`None` 表示完整状态未知。 |

复盘时按 `moves` 的顺序还原棋局。对于 `sample_range` 选中的每一手，训练输入是该手走出前的局面。
