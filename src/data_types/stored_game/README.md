# StoredGame

`PostgreSQLStore` 将每盘已提交的棋谱读成一个 `StoredGame`。 [类型定义](stored_game.py) · [读取接口](../../data/postgresql/README.md#读写接口)

| 字段 | 含义 |
| --- | --- |
| `game_id` | 棋局的唯一 UUID。 |
| `ingest_seq` | 棋局所属池的入库编号，用于分页。 |
| `record_text` | 完整的 MiniZero 棋谱原文。 |
| `source` | 采集批次的 [`SourceRun`](../source_run/README.md)，包含所属池和复盘所需的配置。 |
