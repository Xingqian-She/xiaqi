# 实验棋谱数据库

每个实验在自己的 PostgreSQL 数据目录中使用同一套 [建表 SQL](schema.sql)。[初始化脚本](bootstrap.sh) 准备 `xiaqi` 数据库并校验结构版本。训练、现场采集和导入通过同一个 [PostgreSQLStore](postgresql.py) 读写棋谱。 [数据库 Shell 与命令](../../../PostgreSQL.md)

| 表 | 内容 |
| --- | --- |
| `database_state` | 一行，保存结构版本及最后的 `ingest_seq`。 |
| `source_runs` | 每个采集批次的 UUID、模型 SHA-256、MiniZero 提交号和生效配置。 |
| `games` | 棋局 UUID、批次外键、库内唯一入库编号及原始棋谱。 |

## 读写接口

调用方提供启用 `autocommit` 的 Psycopg 连接。写方法自行开启事务。

| 方法 | 行为 |
| --- | --- |
| `register_source(source)` | 登记或核对不可变的采集批次。 |
| `append_game(game_id, source_run_id, record_text)` | 提交一盘棋谱并返回 `ingest_seq`。 |
| `fetch_games_before(before_seq, limit)` | 降序读取比游标更旧的棋局，传入 `None` 时从最新一盘开始。 |
| `fetch_games_after(after_seq, limit)` | 升序读取编号更大的棋局，传入 `0` 时从头开始。 |

读取合并棋局和采集来源，返回 `StoredGame`。分页游标使用非负整数，`limit` 使用正整数。 [采集批次字段](../../data_types/source_run/README.md) · [棋局字段](../../data_types/stored_game/README.md)

## 事务与重试

逐盘写入时锁住状态行，直到分配编号和棋谱写入一并提交。回填与增量读取用同一个编号，后提交的棋局会有更大的编号。同一批次 UUID 和棋局 UUID 携带相同内容重试时返回原有记录。复用 UUID 时内容不同会报告冲突。来源登记和每盘棋分别提交，可保留中断前已完成的写入。 [增量复盘](../../../PostgreSQL.md#内存窗口与采样)

## 测试

在临时实验的数据库 Shell 中使用 `XIAQI_TEST_DATABASE_URL` 连接。集成用例采用各自的测试 schema，覆盖事务重试、并发编号、外键和分页。

```bash
XIAQI_TEST_DATABASE_URL="$XIAQI_DATABASE_URL" python -m unittest src.data.postgresql.postgresql_test
```
