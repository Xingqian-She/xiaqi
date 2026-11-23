# 自对弈训练指南

## 检查开发环境

`dev` 镜像提供 PyTorch、CUDA、MiniZero 构建依赖、PostgreSQL 和 TensorBoard。仓库挂载到容器的 `/workspace`，源码改动可直接在容器内使用。首次运行按首页步骤创建数据卷并构建镜像、引擎和 Python 绑定。 [联调快跑](../README.md#联调快跑) · [Dockerfile](../docker/Dockerfile)

在宿主机的仓库根目录运行下面的命令。环境脚本检查 GPU、cuDNN 和编译工具，实际执行 CUDA 张量运算，并比较 Inductor 与 eager 的前向结果和梯度。首次编译需要等待，检查失败会返回非零退出码。Inductor、Triton 和 CUDA 的缓存写在容器的 `/tmp`，临时容器退出后清理。 [检查脚本](../scripts/check_env.py)

```powershell
docker compose config --quiet
docker compose run --rm -T dev python scripts/check_env.py
```

构建步骤访问软件源需要代理时，将以下命令中的 `7897` 改为本机代理端口。

```powershell
docker compose build --build-arg http_proxy=http://host.docker.internal:7897 --build-arg https_proxy=http://host.docker.internal:7897 dev
```

## 选择训练配置

随仓配置均使用 9×9 棋盘、7 目贴目、`positional` 劫规则和 GPU 自对弈。联调用于检验采集、复盘、更新和导出的完整流程。本机实验缩短采集轮次，并使用 EMA 权重自对弈。

| 设置 | [联调配置](../configs/training/go_9x9_smoke.toml) | [首版训练配置](../configs/training/go_9x9_az.toml) | [本机 EMA 实验](../configs/training/go_9x9_local_ema.toml) |
| --- | --- | --- | --- |
| 棋谱池 `[run].pool_id` | `smoke-go-9x9` | `train-go-9x9-az` | `train-go-9x9-local-ema` |
| Learner 设备 `[learner].device` | `cpu` | `cuda` | `cuda` |
| 编译训练前向 `[learner].compile_model` | `false` | `true` | `false` |
| EMA `[learner].ema_enabled` | `false` | `false` | `true`，`tau = 0.05` |
| Transformer 宽度 / Encoder 层数 | 8 / 1 | 384 / 2 | 512 / 4 |
| 每手 MCTS 搜索预算 `[alphazero].num_simulations` | 2 | 200 | 256 |
| 自对弈并行盘数 `[selfplay].num_parallel_games` | 1 | 32 | 32 |
| 每轮采集盘数 `[schedule].games_per_round` | 2 | 2,000 | 40 |
| 开训新增棋局门槛 `[schedule].min_new_games_before_train` | 2 | 0 | 40 |
| 每次更新样本数 `[learner].batch_size` | 2 | 1,024 | 1,024 |
| Mini batch 样本数 `[learner].mini_batch_size` | 2 | 1,024 | 512 |
| 开训样本门槛 `[learner].min_train_samples` | 4 | 100,000 | 2,000 |
| 经验池保留目标 `[learner].target_samples` | 64 | 4,000,000 | 500,000 |
| 每轮更新步数 `[schedule].train_steps_per_round` | 2 | 200 | 4 |
| 每次启动的最大轮数 `[schedule].max_rounds` | 2 | 300 | 100,000 |
| 训练检查点保存间隔 `[schedule].checkpoint_every_rounds` | 1 轮 | 1 轮 | 10 轮 |

经验池保存在 CPU 内存中。9×9 的每个样本包含 `18 × 9 × 9` 个特征值、82 个策略值和 1 个价值，三个 `float32` 数组共占 6,164 字节。50 万和 400 万样本的数组分别约占 **2.9 GiB** 和 **23 GiB** 主机内存，进程、整盘保留和组批还需内存。`target_samples` 是按整盘淘汰的保留目标。 [样本形状](../src/data_types/training_samples/README.md) · [经验池实现](../src/core/learner/replay_buffer/README.md)

修改配置时，搜索预算和并行盘数影响采集开销，`batch_size` 控制每次更新样本数，`mini_batch_size` 控制每次前向和反向的样本数。须满足 `batch_size <= min_train_samples <= target_samples`，并且 mini batch 能整除批大小。修改棋盘、贴目或劫规则时使用对应的 `pool_id`。 [Learner 配置](../src/data_types/learner_config/README.md) · [池与采集批次](../PostgreSQL.md#postgresql-存储与接口)

## 启动训练

构建引擎和 Go 版 Python 绑定后，在宿主机的仓库根目录启动联调。

```powershell
docker compose run --rm -T dev bash scripts/train.sh configs/training/go_9x9_smoke.toml
```

运行首版训练配置时，改用下面的命令。

```powershell
docker compose run --rm -T dev bash scripts/train.sh configs/training/go_9x9_az.toml
```

本机 EMA 实验使用独立棋谱池。累计至少 40 盘新棋局成功复盘并产出样本、池内至少有 2,000 个有效位置时执行训练。

```powershell
docker compose run --rm -T dev bash scripts/train.sh configs/training/go_9x9_local_ema.toml
```

`train.sh` 在默认连接上创建缺失的 PostgreSQL 角色、`xiaqi` 库和表，设置 `minizero_py` 的搜索路径，再调用 Python 入口。向容器传入 `XIAQI_DATABASE_URL` 时，脚本直接使用该连接，表由调用方准备。全新运行从初始模型导出首轮采集权重。 [启动脚本](../scripts/train.sh) · [Python 入口](../scripts/train.py) · [初始化数据库](../PostgreSQL.md#初始化数据库)

## 运行产物与进度

终端打印 `run_id` 和容器内的 `run_dir`。随仓配置对应宿主机的 `runs/<run_id>/`。

| 产物 | 用途 |
| --- | --- |
| `config.toml`、`minizero_base.cfg` | 本次使用的训练配置快照和 MiniZero 基础配置副本。 |
| `run.json` | 运行 ID、启动时间、配置来源、初始权重来源和续训路径。 |
| `events.jsonl` | 逐条记录采集、训练、跳过、中断、导出和结束事件。 |
| `tensorboard/` | 按训练步记录损失，按采集轮记录棋局统计、池规模和耗时。 |
| `checkpoints/step-*.pt` | 供 `--resume` 使用的训练模型、优化器、已完成步数和开启时的 EMA 权重。 |
| `selfplay_models/initial.pt` | 使用 `[selfplay].initial_model` 作为首轮权重时保存的输入副本。 |

控制器在容器的 `/dev/shm` 中为本次运行创建临时目录。每轮自对弈都用同一路径的 `model.pt`，新推理权重在下一轮采集前覆盖文件，临时目录随运行退出清理。棋谱来源的 `model_sha256` 区分不同模型版本。 [采集来源](../src/data_types/source_run/README.md)

训练检查点按采集轮号定期保存。跳过训练的轮次也计入间隔，有新更新时才写入。预训练完成时保存一次，正常结束时补存当前尚未保存的训练更新。本机 EMA 配置每 10 轮保存一次，按两轮本机验收的平均耗时估算约 69 分钟。 [检查点格式](../src/core/learner/checkpoint/README.md)

联调在两轮中各采集两盘。通过复盘且达到两项开训门槛时，每轮更新两步，总计入库四盘、更新四步。此时 `run_finished` 的 `rounds`、`games_committed` 和 `train_steps` 分别为 `2`、`4` 和 `4`。若出现 `training_skipped`，查看事件中的样本数、新增有效棋局数与对应门槛。运行失败时看 `run_failed` 的 `phase` 和 `error`。 [训练控制器](../src/control/training_controller/README.md) · [日志指标口径](../src/control/run_log/README.md#tensorboard-指标)

## TensorBoard

构建 `dev` 镜像后，在宿主机的仓库根目录启动查看器。它只读挂载默认的 `runs/` 目录。 [打开 TensorBoard](http://127.0.0.1:6006)

```powershell
docker compose up -d tensorboard
```

用 `docker compose stop tensorboard` 停止查看器。

## 续训

续训使用原运行目录的 `config.toml` 和一个已保存的训练检查点。保留快照引用的 `minizero_base.cfg`，使用存有原棋谱池的数据库。以下以联调第 4 步的检查点为例，将变量替换为终端打印的 `run_id`。

```powershell
$xiaqiRunId = '替换为原运行的 run_id'
docker compose run --rm -T dev bash scripts/train.sh "runs/$xiaqiRunId/config.toml" `
  --resume "runs/$xiaqiRunId/checkpoints/step-000004.pt"
```

控制器恢复模型、优化器和训练步数，再从同一个 `pool_id` 回填经验池。它创建新的运行目录，采集轮号从 1 开始，`max_rounds` 设定本次启动的轮数。联调续训若再完成四次更新，末尾累计步数为 8。

用 `Ctrl+C` 可以中断运行。训练中的更新步会先完成，采集中的 MiniZero 子进程会先收尾。控制器保存最新一致的训练状态，并在终端打印可用于 `--resume` 的路径。最新检查点写入失败时，终端会报告原因，并在已有有效检查点时显示其路径。已提交的棋谱可供续训时回填。 [训练检查点](../src/core/learner/checkpoint/README.md)
