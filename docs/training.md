# 自对弈训练指南

## 检查开发环境

`dev` 镜像提供 PyTorch、CUDA、MiniZero 构建依赖、PostgreSQL 和 TensorBoard。仓库挂载到容器的 `/workspace`，源码改动可直接在容器内使用。首次运行按首页步骤构建镜像、引擎和 Python 绑定。 [联调快跑](../README.md#联调快跑) · [Dockerfile](../docker/Dockerfile)

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

随仓配置均使用 9×9 棋盘、7 目贴目、`positional` 劫规则和 GPU 自对弈。联调用于检验采集、复盘、更新和导出的完整流程。本机实验使用 EMA 权重自对弈，表中列出第 1300 步后的续训设置。

| 设置 | [联调配置](../runs/smoke/config.toml) | [首版训练配置](../runs/go-9x9-az/config.toml) | [本机 EMA 实验](../runs/go-9x9-ema/config.toml) |
| --- | --- | --- | --- |
| 实验名 | `smoke` | `go-9x9-az` | `go-9x9-ema` |
| Learner 设备 `[learner].device` | `cpu` | `cuda` | `cuda` |
| 编译训练前向 `[learner].compile_model` | `false` | `true` | `false` |
| EMA `[learner].ema_enabled` | `false` | `false` | `true`，`tau = 0.05` |
| Transformer 宽度 / Encoder 层数 | 8 / 1 | 384 / 2 | 512 / 4 |
| 每手 MCTS 搜索预算 `[alphazero].num_simulations` | 2 | 200 | 400 |
| 自对弈并行盘数 `[selfplay].num_parallel_games` | 1 | 32 | 32 |
| 每轮采集盘数 `[schedule].games_per_round` | 2 | 2,000 | 160 |
| 开训新增棋局门槛 `[schedule].min_new_games_before_train` | 2 | 0 | 40 |
| 每次更新样本数 `[learner].batch_size` | 2 | 1,024 | 4,096 |
| Mini batch 样本数 `[learner].mini_batch_size` | 2 | 1,024 | 512 |
| 开训样本门槛 `[learner].min_train_samples` | 4 | 100,000 | 4,096 |
| 经验池保留目标 `[learner].target_samples` | 64 | 4,000,000 | 1,000,000 |
| 每轮更新步数 `[schedule].train_steps_per_round` | 2 | 200 | 6 |
| 每次启动的最大轮数 `[schedule].max_rounds` | 2 | 300 | 100,000 |
| 训练检查点保存间隔 `[schedule].checkpoint_every_rounds` | 1 轮 | 1 轮 | 10 轮 |

早期 EMA 训练线使用独立的实验名 `go-9x9-ema-early`。它保留第 12 步对应的配置，每手搜索 256 次。后续训练线使用 `go-9x9-ema`。 [早期 EMA 配置](../runs/go-9x9-ema-early/config.toml) · [早期训练记录](training-notes/2026-11-23-go-9x9-local-ema.md)

经验池保存在 CPU 内存中。9×9 的每个样本包含 `18 × 9 × 9` 个特征值、82 个策略值和 1 个价值，三个 `float32` 数组共占 6,164 字节。100 万和 400 万样本的数组分别约占 **5.7 GiB** 和 **23 GiB** 主机内存，进程、整盘保留和组批还需内存。`target_samples` 是按整盘淘汰的保留目标。 [样本形状](../src/data_types/training_samples/README.md) · [经验池实现](../src/core/learner/replay_buffer/README.md)

修改配置时，搜索预算和并行盘数影响采集开销，`batch_size` 控制每次更新样本数，`mini_batch_size` 控制每次前向和反向的样本数。须满足 `batch_size <= min_train_samples <= target_samples`，并且 mini batch 能整除批大小。更换棋盘、贴目、劫规则或模型结构时使用新的实验目录。 [Learner 配置](../src/data_types/learner_config/README.md) · [棋谱与采集批次](../PostgreSQL.md#postgresql-存储与接口)

## 启动训练

构建引擎和 Go 版 Python 绑定后，在宿主机的仓库根目录启动联调。

```powershell
.\scripts\run.ps1 smoke -New
```

运行首版训练配置时，改用下面的命令。

```powershell
.\scripts\run.ps1 go-9x9-az -New
```

自建实验时，将 `config.toml` 和 `minizero_base.cfg` 放进 `runs/<实验名>/`，再用该实验名启动。启动器从脚本位置定位仓库，挂载本实验的数据库目录并初始化结构。已有检查点时使用 `-Resume`。 [启动脚本](../scripts/run.ps1) · [Python 入口](../scripts/train.py) · [初始化数据库](../PostgreSQL.md#初始化数据库)

## 运行产物与进度

终端打印本次启动的 `run_id` 和固定的 `run_dir`。后者对应宿主机的 `runs/<实验名>/`。

| 产物 | 用途 |
| --- | --- |
| `config.toml`、`minizero_base.cfg` | 实验的活动训练配置和 MiniZero 基础配置。 |
| `postgresql/` | 本实验持久化的棋谱、采集来源和入库编号。 |
| `events.jsonl` | 追加每次启动实际使用的配置和恢复点，以及采集、训练、中断、导出和结束事件。 |
| `tensorboard/train/`、`tensorboard/rounds/` | 分别按训练步和实验累计采集轮记录指标。 |
| `checkpoints/step-*.pt` | 供 `-Resume` 使用的模型、优化器、已完成步数和启用时的 EMA 权重。 |
| `selfplay_models/initial.pt` | 使用 `[selfplay].initial_model` 作为首轮权重时保存的输入副本。 |

控制器在容器的 `/dev/shm` 中固定本次使用的 MiniZero 基础配置。每轮自对弈都用同一路径的 `model.pt`，新推理权重在下一轮采集前覆盖文件，临时目录随运行退出清理。棋谱来源的 `model_sha256` 区分不同模型版本。 [采集来源](../src/data_types/source_run/README.md)

训练检查点按采集轮号定期保存。跳过训练的轮次也计入间隔，有新更新时才写入。预训练完成时保存一次，正常结束时补存当前尚未保存的训练更新。本机 EMA 配置每 10 轮保存一次。 [检查点格式](../src/core/learner/checkpoint/README.md)

联调在两轮中各采集两盘。通过复盘且达到两项开训门槛时，每轮更新两步，总计入库四盘、更新四步。此时 `run_finished` 的 `rounds`、`games_committed` 和 `train_steps` 分别为 `2`、`4` 和 `4`。若出现 `training_skipped`，查看事件中的样本数、新增有效棋局数与对应门槛。运行失败时看 `run_failed` 的 `phase` 和 `error`。 [训练控制器](../src/control/training_controller/README.md) · [日志指标口径](../src/control/run_log/README.md#tensorboard-指标)

## TensorBoard

在仓库根目录启动查看器。默认只读挂载联调实验的日志。要查看 EMA 实验，先设置下面的环境变量。 [打开 TensorBoard](http://127.0.0.1:6006)

```powershell
$env:XIAQI_TENSORBOARD_DIR = './runs/go-9x9-ema/tensorboard'
docker compose up -d tensorboard
```

用 `docker compose stop tensorboard` 停止查看器。

## 续训

续训使用实验目录的活动配置和现有 PostgreSQL 库。控制器取得实验训练锁后，按数字步数选择最新检查点，直到训练结束才释放锁。同一个实验的第二个训练进程会在采集和写日志之前退出。以下命令适用于已有检查点的联调实验和迁移后的本机 EMA 实验。

```powershell
.\scripts\run.ps1 smoke -Resume
.\scripts\run.ps1 go-9x9-ema -Resume
```

控制器恢复模型、优化器、EMA 和训练步数，从本实验的库回填内存经验池。待训棋局计数和事件中的采集轮号从本次启动重新计算，`max_rounds` 设定本次启动的轮数。联调从第 4 步续训并再完成四次更新后，原目录会得到第 8 步检查点。

用 `Ctrl+C` 可以中断运行。训练中的更新步会先完成，采集中的 MiniZero 子进程会先收尾。控制器保存最新一致的训练状态，并打印可用于 `-Resume` 的检查点路径。最新检查点写入失败时，终端会报告原因，并在已有有效检查点时显示其路径。已提交的棋谱可供续训时回填。 [训练检查点](../src/core/learner/checkpoint/README.md)
