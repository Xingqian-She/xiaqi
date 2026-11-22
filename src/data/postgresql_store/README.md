# PostgreSQL 棋谱存储

`PostgreSQLStore` 登记 MiniZero 自对弈的来源信息，并将每盘棋的原始 `SelfPlay` 消息逐局写入 PostgreSQL。采集脚本负责筛选完整消息和生成棋局 ID。复盘时可将库里的 `record_text` 交给 `parse_record()`。 [实现代码](postgresql_store.py) · [自对弈采集](../minizero_selfplay/README.md) · [棋谱解析](../minizero_record/README.md)

## 数据表

| 表 | 内容 |
| --- | --- |
| `pools` | 每个经验池一行，保存 `pool_id` 和最后分配的入库编号 `last_ingest_seq`，计数从 `0` 开始。 |
| `source_runs` | 每个采集批次一行。`SourceRun` 提供批次 UUID、所属池、模型 SHA-256、MiniZero commit 和生效配置全文。 |
| `games` | 每盘棋一行，保存全表唯一的 `game_id`、所属池、采集批次、池内编号 `ingest_seq` 和棋谱原文 `record_text`。 |

`games` 的联合外键 `(pool_id, source_run_id)` 保证棋局与所属批次引用同一个池。`(pool_id, ingest_seq)` 在表内唯一，不同池可以各自从 `1` 编号。 [建表 SQL](schema.sql) · [存储与增量读取设计](../../../PostgreSQL.md#postgresql-存储与接口)

## 调用

从仓库根目录执行建表 SQL。数据库创建和 `XIAQI_DATABASE_URL` 的设置见项目根目录说明。 [数据库准备](../../../README.md#自对弈棋谱入库)

```bash
psql "$XIAQI_DATABASE_URL" -v ON_ERROR_STOP=1 -1 -f src/data/postgresql_store/schema.sql
```

用 Psycopg 3 打开 `autocommit=True` 的连接。调用存储方法时，连接应处于事务之外，由模块为每次调用开启事务。示例中的 `source` 是本次采集的 `SourceRun`，`records` 是按顺序筛出的完整棋谱。 [来源字段](../../data_types/source_run/README.md) · [采集脚本](../../../scripts/collect_selfplay.py)

```python
import os
from uuid import uuid5

import psycopg

from src.data.postgresql_store.postgresql_store import PostgreSQLStore

with psycopg.connect(os.environ["XIAQI_DATABASE_URL"], autocommit=True) as connection:
    store = PostgreSQLStore(connection)
    store.register_source(source)
    for index, record_text in enumerate(records, start=1):
        game_id = uuid5(source.source_run_id, str(index))
        ingest_seq = store.append_game(source.pool_id, game_id, source.source_run_id, record_text)
        print(ingest_seq)
```

## 事务与重试

- `register_source(source)` 在同一个事务里按需创建池、插入来源并回查元数据。同一批次 ID 和相同元数据可重复登记。复用批次 ID 时若池、权重哈希、引擎提交号或配置全文有变化，会抛出 `IngestConflictError`。
- `append_game(...)` 先用 `FOR NO KEY UPDATE` 锁住所属池的计数行，再尝试按 `last_ingest_seq + 1` 插入。新棋局及计数更新在同一个事务里提交。同池的并发写入会等待前一个事务结束，保证较大的编号不会先提交，供读取方用 `ingest_seq > 上次编号` 拉取新棋局。
- 同一个 `game_id` 重试时，池、批次和棋谱原文完全一致就返回原编号，计数只在插入新棋局时推进。任何一项不同都会抛出 `IngestConflictError`。用相同批次 UUID 和原顺序重读完整记录时，`uuid5()` 会为每盘重建同一个 ID。

来源登记和每盘写入分别提交。中途失败时，之前提交的棋局仍保存在库里。 [写入实现](postgresql_store.py)

## 测试

在已设置 `XIAQI_DATABASE_URL` 的开发容器中运行。每个用例会创建独立的 schema 并在结束时清理，需要连接用户拥有创建 schema 的权限。用例检查原文写入、重复提交、来源和池的冲突处理以及并发编号分配。未设置 `XIAQI_TEST_DATABASE_URL` 时跳过。 [测试代码](postgresql_store_test.py)

```bash
XIAQI_TEST_DATABASE_URL="$XIAQI_DATABASE_URL" \
  python -m unittest -v src.data.postgresql_store.postgresql_store_test
```
