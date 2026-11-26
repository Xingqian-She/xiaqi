# 自对弈棋谱存储与内存经验池

每个实验的 PostgreSQL 数据保存在 `runs/<实验名>/postgresql/`，使用共用的表结构。训练控制器从本实验的库回填近期棋谱，每轮采集后增量复盘，Learner 从 CPU 内存中的经验池抽样。 [数据模块](src/data/README.md) · [训练指南](docs/training.md)

## 初始化数据库

在 PowerShell 中用实验名启动训练或数据库 Shell。启动器为该实验挂载数据目录，启动 PostgreSQL，准备 `xiaqi` 库。首次初始化使用共用的建表 SQL。数据库 Shell 可用于采集、导入与查询。 [启动器](scripts/run.ps1) · [建表 SQL](src/data/postgresql/schema.sql)

```powershell
.\scripts\run.ps1 smoke -Shell
```

容器中可用 `psql "$XIAQI_DATABASE_URL"` 查询当前实验。Shell 的数据库与实验名绑定。在 Shell 内启动训练时，配置文件须属于这个实验。已有检查点时，`-Shell` 和 `-Resume` 都要求原实验库存在。切换到同一实验的训练或续训前先退出 Shell。换机器或升级 PostgreSQL 大版本时，用 `pg_dump -Fc` 备份本实验的库，再用 `pg_restore` 恢复到空数据库。 [容器入口](docker/dev-entrypoint.py)

## 采集与导入

在所选实验的数据库 Shell 中执行以下命令。现场采集使用 MiniZero Go 引擎、配套的 `.commit` 文件和模型。脚本打印采集批次 UUID、每盘已提交的 `game_id` 和 `ingest_seq`。 [构建引擎](third_party/README.md#构建) · [公开权重](third_party/README.md#公开权重)

```bash
python -m scripts.collect_selfplay --games 1 \
  --model models/minizero/go_9x9_az_n200.pt \
  --conf-str 'program_quiet=true:zero_num_parallel_games=1:zero_num_threads=1:actor_num_simulation=2:zero_disable_resign_ratio=1'
```

`--record-file` 从已有日志筛选完整的 `SelfPlay true … #` 消息。将三个变量设为原批次的 UUID、权重 SHA-256 和 MiniZero 完整提交号。`effective.cfg` 使用采集时的生效配置，可用引擎的 `-gen` 参数导出。 [棋谱消息与生效配置](src/data/minizero_selfplay/README.md)

```bash
python -m scripts.collect_selfplay --record-file record.txt \
  --source-run-id "$source_run_id" --model-sha256 "$model_sha256" \
  --minizero-commit "$minizero_commit" --effective-config effective.cfg
```

首次导入时保存这批使用的 UUID。导入中断后，在同一实验中用原批次 UUID、来源元数据和棋谱顺序重跑，已提交的棋局会返回原入库编号。查询本实验最近入库的棋局可用下面的命令。 [事务与重试](src/data/postgresql/README.md#事务与重试)

```bash
psql "$XIAQI_DATABASE_URL" -c 'SELECT ingest_seq, game_id, source_run_id FROM games ORDER BY ingest_seq DESC LIMIT 5'
```

## PostgreSQL 存储与接口

`source_run_id` 标识一个采集批次，关联模型 SHA-256、MiniZero 提交号和生效配置。每盘棋保留原始 `record_text` 和唯一的 `game_id`。入库事务锁住库状态行，同步分配编号和写入棋谱，使增量读取得到按提交顺序排列的 `ingest_seq`。 [表结构与读写接口](src/data/postgresql/README.md)

## 内存窗口与采样

1. **回填。** 启动时按 `ingest_seq` 从新到旧读取和复盘，直到达到 `[learner].target_samples` 或读完数据库。第一页最大编号作为增量游标，空库从 `0` 开始。
2. **增量复盘。** 每轮采集后按编号升序读取至本轮最后提交的编号。通过规则与复盘校验、且产出样本的棋局按整盘入池。无效棋谱记录 `game_skipped` 并推进游标，数据库或复盘环境故障结束本次运行。
3. **整盘淘汰和采样。** 剩余样本达到保留目标时淘汰最旧一盘，历史棋谱仍保存在数据库。经验池达到 `min_train_samples` 后，Learner 从全池均匀、无放回地抽取训练批次。

[棋谱解析](src/data/minizero_record/README.md) · [复盘校验](src/data/minizero_replay/README.md#校验) · [内存经验池](src/core/learner/replay_buffer/README.md)
