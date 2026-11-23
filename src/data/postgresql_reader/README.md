# PostgreSQL 棋谱读取

`PostgreSQLReader` 按经验池分页读取已提交的原始棋谱。每条 `StoredGame` 包含 `game_id`、`ingest_seq`、`record_text` 和 `source`。一次查询同时取得棋谱和关联的 `SourceRun`，其中 `effective_config` 可供训练入口确定复盘规则。 [实现代码](postgresql_reader.py) · [返回字段](../../data_types/stored_game/README.md) · [采集批次字段](../../data_types/source_run/README.md)

| 接口 | 读取顺序 |
| --- | --- |
| `fetch_games_before(pool_id, before_seq, limit)` | 按编号降序读取严格小于 `before_seq` 的棋局。传入 `None` 时从最新一盘开始。 |
| `fetch_games_after(pool_id, after_seq, limit)` | 按编号升序读取严格大于 `after_seq` 的棋局。传入 `0` 时从最早一盘开始。 |

游标为非负整数，`limit` 为正整数。两个接口均返回 `list[StoredGame]`，无匹配棋局时返回 `[]`。已有 `(pool_id, ingest_seq)` 索引支持按池查找和双向排序。 [建表 SQL](../postgresql_store/schema.sql) · [PostgreSQL 索引排序](https://www.postgresql.org/docs/current/indexes-ordering.html)

## 调用

在已建表的开发环境中，用装有 Psycopg 3 的 Python 运行示例。由调用方管理 `autocommit=True` 的连接，保留默认的元组行格式，在事务外读取每一页。 [数据库准备](../../../PostgreSQL.md#初始化数据库) · [Psycopg 事务说明](https://www.psycopg.org/psycopg3/docs/basic/transactions.html)

```python
import os

import psycopg

from src.data.postgresql_reader.postgresql_reader import PostgreSQLReader

with psycopg.connect(os.environ["XIAQI_DATABASE_URL"], autocommit=True) as connection:
    reader = PostgreSQLReader(connection)
    pool_id = "official-go-9x9"
    newest = reader.fetch_games_before(pool_id, None, limit=64)
    after_seq = newest[0].ingest_seq if newest else 0
    if newest:
        older = reader.fetch_games_before(pool_id, newest[-1].ingest_seq, limit=64)
    updates = reader.fetch_games_after(pool_id, after_seq, limit=64)
```

训练入口将 `game.record_text` 交给 `parse_record()`，复盘校验后按盘入池。回填继续用上一页最后一盘的编号向前翻页，增量从启动第一页的最大编号起算，可读到回填期间的新提交。增量游标推进到最后一条已处理记录。查询失败时异常传给调用方，重试时可继续使用原游标。 [棋谱解析](../minizero_record/README.md) · [启动与增量流程](../../../PostgreSQL.md#内存窗口与采样)

## 测试

从仓库根目录运行。集成用例使用独立的测试 schema，覆盖翻页边界、来源匹配、池隔离，以及回填期间的新提交。未配置 `XIAQI_TEST_DATABASE_URL` 时运行参数校验用例。 [测试代码](postgresql_reader_test.py)

```bash
XIAQI_TEST_DATABASE_URL="$XIAQI_DATABASE_URL" \
  python -m unittest -v src.data.postgresql_reader.postgresql_reader_test
```
