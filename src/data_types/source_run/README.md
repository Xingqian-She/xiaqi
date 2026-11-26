# SourceRun

`SourceRun` 保存一次 MiniZero 自对弈采集的来源，由 `PostgreSQLStore.register_source()` 登记。同一批次重试时复用批次 UUID 和原始元数据。 [类型定义](source_run.py) · [来源登记](../../data/postgresql/README.md#事务与重试) · [库内棋谱](../stored_game/README.md)

| 字段 | 含义 |
| --- | --- |
| `source_run_id` | 采集批次 UUID。 |
| `model_sha256` | 用于采集的模型文件 SHA-256。 |
| `minizero_commit` | 用于采集的 MiniZero 提交号。 |
| `effective_config` | 应用配置覆盖后的 MiniZero 配置全文。 |
