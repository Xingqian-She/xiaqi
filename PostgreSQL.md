# 自对弈棋谱存储与内存经验池

PostgreSQL 保存 MiniZero 的原始棋谱和采集来源。训练控制器启动时按池回填近期棋谱，每轮采集后增量复盘，Learner 从 CPU 内存中的样本取训练批次。 [数据模块](src/data/README.md) · [训练指南](docs/training.md)

## 初始化数据库

`dev` 容器启动时会启动 PostgreSQL，数据保存在 `xiaqi-postgresql-data` 卷中。同一时间由一个 `dev` 容器持有这个卷。切换容器前先退出原容器，也可以直接在已运行的容器里执行相关命令。换机器或升级 PostgreSQL 大版本时，可用 `pg_dump -Fc` 备份，再用 `pg_restore` 恢复。 [容器入口](docker/dev-entrypoint.py) · [Compose 配置](compose.yaml)

用 `scripts/train.sh` 启动训练时，脚本会初始化默认库。单独使用采集或导入命令时，先按首页准备开发镜像和数据卷，再从宿主机进入容器。 [联调快跑](README.md#联调快跑) · [训练启动](docs/training.md#启动训练)

```powershell
docker compose run --rm dev
```

在容器终端中，分别在角色或数据库缺失时执行前两行。通过 Unix socket 连接时，默认使用容器当前用户名进行 peer 认证。已有默认库时，从设置连接串开始即可，建表 SQL 可重复运行。 [建表 SQL](src/data/postgresql_store/schema.sql)

```bash
runuser -u postgres -- createuser "$(id -un)"
runuser -u postgres -- createdb --owner="$(id -un)" xiaqi
export XIAQI_DATABASE_URL='dbname=xiaqi host=/var/run/postgresql'
psql "$XIAQI_DATABASE_URL" -v ON_ERROR_STOP=1 -1 -f src/data/postgresql_store/schema.sql
```

管理员可用 `runuser -u postgres -- psql -d postgres` 连接，输入 `\q` 退出。容器内的 TCP 监听地址为 `localhost:5432`。

## 采集与导入

### 现场采集

准备 MiniZero Go 引擎、配套的 `.commit` 文件和公开权重后，在已设置连接串的 GPU `dev` 容器中采集一盘。脚本会打印批次 UUID，以及每盘已提交的 `game_id` 和 `ingest_seq`。 [构建引擎](third_party/README.md#构建) · [公开权重](third_party/README.md#公开权重) · [采集入口](scripts/collect_selfplay.py)

```bash
python -m scripts.collect_selfplay --pool-id official-go-9x9 --games 1 \
  --model models/minizero/go_9x9_az_n200.pt \
  --conf-str 'program_quiet=true:zero_num_parallel_games=1:zero_num_threads=1:actor_num_simulation=2:zero_disable_resign_ratio=1'
```

### 导入输出文件

`--record-file` 从已有日志筛选完整的 `SelfPlay true … #` 消息。执行前将三个变量分别设为这批棋谱的 UUID、权重 SHA-256 和 MiniZero 完整提交号。`effective.cfg` 使用这批采集的生效配置，运行引擎时可用相同参数加 `-gen effective.cfg` 导出。 [消息筛选与生效配置](src/data/minizero_selfplay/README.md)

```bash
python -m scripts.collect_selfplay --pool-id official-go-9x9 --record-file record.txt \
  --source-run-id "$source_run_id" --model-sha256 "$model_sha256" \
  --minizero-commit "$minizero_commit" --effective-config effective.cfg
```

首次导入尚无 UUID 的批次时，生成一个 UUID 并保存。导入中断后，用同一池、批次 UUID、来源元数据和原始棋谱顺序重跑，已经写入的棋局会返回原入库编号。 [事务与重试](src/data/postgresql_store/README.md#事务与重试)

采集或导入后，可在容器内查询这个池最近入库的棋局。

```bash
psql "$XIAQI_DATABASE_URL" -c "SELECT ingest_seq, game_id, source_run_id FROM games WHERE pool_id = 'official-go-9x9' ORDER BY ingest_seq DESC LIMIT 5"
```

## PostgreSQL 存储与接口

`pool_id` 区分经验池。同一池使用一致的棋盘尺寸、贴目和劫规则。`source_run_id` 标识一次采集批次，关联模型 SHA-256、MiniZero 提交号和生效配置。每轮采集会登记新的批次，同一次训练使用同一池。每盘棋保留原始 `record_text` 和唯一的 `game_id`，`ingest_seq` 是池内的入库顺序。

入库时锁住池计数行，在同一事务内分配编号并写入棋谱，使同池的提交顺序与编号一致。控制器用这个编号分页和更新增量游标。表结构、Python 接口和数据库测试见各模块说明。 [棋谱写入](src/data/postgresql_store/README.md) · [棋谱读取](src/data/postgresql_reader/README.md)

## 内存窗口与采样

1. **回填。** 启动时按 `ingest_seq` 从新到旧读取与复盘，直到达到 `[learner].target_samples` 或读完池。第一页最大编号作为增量游标，空池从 `0` 开始。
2. **增量复盘。** 每轮采集结束后，控制器按编号升序读取至本轮最后提交的编号。通过规则与复盘校验且产出样本的棋局按整盘入池。无效棋谱记录 `game_skipped` 并推进游标，数据库或复盘环境故障结束本次运行。
3. **整盘淘汰和采样。** 剩余样本够保留目标时淘汰最旧一盘，历史棋谱仍保存在数据库。经验池达到 `min_train_samples` 后，Learner 从全池均匀、无放回地抽取训练批次。

[棋谱解析](src/data/minizero_record/README.md) · [复盘校验](src/data/minizero_replay/README.md#校验) · [控制器流程](src/control/training_controller/README.md) · [内存经验池](src/core/learner/replay_buffer/README.md)
