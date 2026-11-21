# 自对弈棋谱存储与内存经验池

## 首版需求

MiniZero 用 `go_9x9_az_n200.pt` 生成完整棋谱，PostgreSQL 持久化原始文本和采集信息。训练程序启动时把最近的有效棋谱复盘成内存经验池，运行中增量处理新棋局。每个训练批次直接从内存池取样。 [现有自对弈入口](third_party/README.md#自对弈) · [现有复盘接口](src/data/minizero_replay/README.md)

- 首版使用 9×9 围棋、7 目贴目和 `positional` 劫规则。同一经验池使用相同的规则和样本格式。 [9×9 配置](configs/minizero/go_9x9_az.cfg)
- 每个独立经验池默认保留至少 `500_000` 条通过复盘校验的训练位置。按盘加入和淘汰，按实际样本数计量容量。达到目标后可开始抽取训练批次。
- `pool_id` 标识棋局所属的经验池，`source_run_id` 标识实际生成棋局的采集批次。两套模型可以读取同一官方数据池。各自开始自对弈时使用独立的池。同一模型迭代更新权重时沿用原池，登记新的采集批次。

## PostgreSQL 存储与接口

每盘一条记录。`record_text` 保留完整的 `SelfPlay true … #` 文本。每个采集批次保存权重 SHA-256、MiniZero 提交号和生效配置，包括实际使用的规则和搜索参数。

| 字段 | 用途 |
| --- | --- |
| `game_id` | 采集端为同一盘生成的稳定 ID，写入重试时复用。 |
| `pool_id` | 棋局所属经验池。 |
| `ingest_seq` | 池内递增的入库顺序，用于增量读取和排序。 |
| `source_run_id` | 关联采集批次，供读取方按对应配置复盘。 |
| `record_text` | 供 `parse_record` 读取的原始棋谱。 |

按 `game_id` 保证唯一，按 `(pool_id, ingest_seq)` 建唯一索引。分配编号和写入棋局在同一事务完成。并发写入同一池时锁住该池的计数行再分配编号，保证读取方看到某个编号后，后续提交的棋局编号都更大。普通自增序列在事务提交前就可能分配编号，用它做增量游标会遇到较小编号晚提交的问题。 [PostgreSQL 序列](https://www.postgresql.org/docs/current/functions-sequence.html) · [PostgreSQL 行锁](https://www.postgresql.org/docs/current/explicit-locking.html#LOCKING-ROWS)

以下是采集端和训练程序调用的 Python 接口草案。

| 接口 | 行为 |
| --- | --- |
| `register_source(source_run)` | 登记采集批次及其权重、引擎版本和生效配置。 |
| `append_game(pool_id, game_id, source_run_id, record_text) -> int` | 写入一盘完整棋谱，返回已提交的 `ingest_seq`。同一 ID 和相同内容重试时返回原编号。若同一 ID 的池、来源或内容不一致，报告冲突。 |
| `fetch_games_before(pool_id, before_seq, limit) -> list[StoredGame]` | 按编号降序读取 `ingest_seq < before_seq` 的棋局。`before_seq=None` 时从最新一盘开始。 |
| `fetch_games_after(pool_id, after_seq, limit) -> list[StoredGame]` | 按编号升序读取 `ingest_seq > after_seq` 的棋局，供运行中增量更新。 |

读取返回棋谱文本、编号及复盘所需的来源信息。采集端负责识别完整输出和写入，训练程序串行调用 `parse_record`、`replay_record` 并校验池的规则。通过校验且样本数大于零的棋局才能入池，计数取自 `TrainingSamples.features.shape[0]`。 [棋谱解析](src/data/minizero_record/README.md) · [复盘校验](src/data/minizero_replay/README.md#校验)

## 内存窗口与采样

1. **启动建池。** 从库里按新到旧分页，每盘复盘一次，直到凑满目标条数或读完该池。用第一页的最大编号作为后续增量游标，空池从 `0` 开始。建池期间新提交的棋局在下一次增量更新中读取。内存中按旧到新的顺序保存复盘成功的棋局。
2. **增量更新。** 在训练批次之间分批拉取游标之后的新棋局，通过校验就将整盘加入窗口。遇到格式或棋局校验错误时记录棋局 ID 和原因，并推进游标。数据库连接或复盘环境故障时保留游标，重试尚未处理的记录。
3. **整盘淘汰。** 新棋局加入后，只要 `当前有效样本总数 - 最老一盘的样本数 >= 目标条数`，就释放最老一盘的数组，然后继续检查。第一次就绪后，窗口至少有目标条数，多出的样本少于当前最老一盘的样本数。历史棋谱继续保存在 PostgreSQL 中。
4. **全池抽样。** 直接保留每盘现有的 `TrainingSamples` 数组。每次更新后，用各盘样本数建立累计索引。从整池均匀抽取批内互不重复的样本编号，再定位到所属棋局和盘内位置。将对应的状态、策略和价值组装为同一批传给训练循环。

首版由训练进程持有和更新内存池，在同一进程调用采样接口。若通过 PyTorch `DataLoader` 取样，使用 `num_workers=0`。每次训练只将当前批次传到 GPU。 [PyTorch 单进程数据加载](https://docs.pytorch.org/docs/stable/data.html#single-process-data-loading-default)

| 接口 | 行为 |
| --- | --- |
| `restore(pool_id, target_samples=500_000) -> PoolStatus` | 从 PostgreSQL 建池并初始化增量游标，返回就绪状态。 |
| `refresh(max_games) -> PoolStatus` | 本轮最多处理 `max_games` 盘新棋局，返回更新后的窗口状态。 |
| `sample_batch(batch_size, rng) -> TrainingSamples` | 从已就绪窗口中全池抽样，返回与现有训练接口兼容的数组。池未就绪时报告当前样本数。 |

`PoolStatus` 包含有效样本数、活跃棋局数、是否就绪、窗口最老编号和最后已处理的编号。

## 容量与验收

9×9 的三个 `float32` 数组每条共 `(18 × 9 × 9 + 82 + 1) × 4 = 6,164` 字节。50 万条约 **2.87 GiB**，同时持有两个满池约 **5.74 GiB**。这是样本数组本身的预算，进程运行、整盘超额和组装训练批次还需留余量。按盘持有数组可以直接释放淘汰的棋局，更新采样索引时仅需重算每盘累计样本数。 [现有样本形状](src/data/minizero_replay/README.md#输出)

- 用较小阈值验证启动建池、增量读入、整盘淘汰、两池隔离和同一棋局重试。覆盖启动期间有新局提交、并发写入和校验失败的情况。
- 验证长局和短局里的各个训练位置拥有相同的入选概率，抽出的三个数组行数一致。
- 用官方权重的实际棋谱测量启动复盘耗时、新局的复盘吞吐、采样组批时延和容器内存峰值，再确认 50 万条的运行预算。
