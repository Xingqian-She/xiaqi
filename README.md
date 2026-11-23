# xiaqi

重新开始的下棋 AI 项目，使用 Docker 提供 Python/PyTorch GPU 开发环境。
旧版代码和资料保留在 `Zero-TTT` 分支。

9×9 围棋推理和自对弈见 [MiniZero 构建与运行](third_party/README.md#minizero)。

项目数据类集中在 `src/data_types/`。 [数据类目录](src/data_types/README.md)

## 开发环境

需要 Docker Compose，以及可供 Linux 容器使用的 NVIDIA GPU。
沿用基础镜像 `pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel`，
提供 Python、PyTorch、CUDA、cuDNN 和 Triton。开发镜像还包含 C/C++ 编译工具、MiniZero 的构建依赖、PostgreSQL 19 和 Python 驱动 Psycopg 3。

在仓库根目录创建数据卷、构建并进入容器。重复执行 `docker volume create` 会复用同名卷。

```powershell
docker volume create xiaqi-postgresql-data
docker compose build dev
docker compose run --rm dev
```

如果 Docker Hub 或软件源连接超时，可通过临时代理构建。
以下使用本机当前的代理端口 `7897`，其他机器按实际地址调整：

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
docker compose build --build-arg http_proxy=http://host.docker.internal:7897 --build-arg https_proxy=http://host.docker.internal:7897 dev
```

仓库挂载到 `/workspace`，修改代码无需重建镜像；输入 `exit` 退出容器。

### TensorBoard

构建 `dev` 镜像后，在仓库根目录按需启动查看器。

```powershell
docker compose up -d tensorboard
```

查看器只读挂载仓库的 `runs/` 目录。训练写入事件文件后，可在本机浏览器查看曲线。[打开 TensorBoard](http://127.0.0.1:6006)

用 `docker compose stop tensorboard` 停止查看器。

### PostgreSQL

开发容器启动时会启动 PostgreSQL。在容器的终端中用下面的命令连接，输入 `\q` 退出。

```bash
runuser -u postgres -- psql -d postgres
```

数据库监听容器内的 `localhost:5432`。数据文件保存在独立的 `xiaqi-postgresql-data` 卷中，容器退出或重建后可以继续使用。更换容器前先退出原容器，同一时间由一个开发容器使用该卷。换机器或升级 PostgreSQL 大版本时，可以先用 `pg_dump -Fc` 导出，再用 `pg_restore` 恢复。

### 自对弈棋谱入库

用当前 Dockerfile 构建 `dev` 镜像后，在容器终端里创建项目数据库。首次执行前两行，后续进入容器时设置连接串即可。Unix socket 的 peer 认证使用容器当前用户名。

```bash
runuser -u postgres -- createuser "$(id -un)"
runuser -u postgres -- createdb --owner="$(id -un)" xiaqi
export XIAQI_DATABASE_URL='dbname=xiaqi host=/var/run/postgresql'
psql "$XIAQI_DATABASE_URL" -v ON_ERROR_STOP=1 -1 -f src/data/postgresql_store/schema.sql
```

用 `bash scripts/build_minizero.sh` 构建引擎及其提交号文件，再采集一盘。每盘提交后会打印棋局 ID 和池内入库编号。 [构建与权重](third_party/README.md#构建)

```bash
python -m scripts.collect_selfplay --pool-id official-go-9x9 --games 1 \
  --conf-str 'program_quiet=true:zero_num_parallel_games=1:zero_num_threads=1:zero_disable_resign_ratio=1'
```

导入已有输出文件时，给原采集批次指定一个 UUID，并传入当时的权重 SHA-256、MiniZero 提交号和生效配置文件。中断后可以用同一批次 UUID 和原顺序重新导入。

```bash
python -m scripts.collect_selfplay --pool-id official-go-9x9 --record-file record.txt \
  --source-run-id "$source_run_id" --model-sha256 "$model_sha256" \
  --minizero-commit "$minizero_commit" --effective-config effective.cfg
```

通过 `psql` 查看最近写入的记录。

```bash
psql "$XIAQI_DATABASE_URL" -c 'SELECT pool_id, ingest_seq, game_id FROM games ORDER BY pool_id, ingest_seq DESC LIMIT 5'
```

存储测试会在所连接的数据库内创建临时测试 schema。 [数据设计](PostgreSQL.md)

```bash
XIAQI_TEST_DATABASE_URL="$XIAQI_DATABASE_URL" \
  python -m unittest -v src.data.postgresql_store.postgresql_store_test
```

### 自对弈训练

在 `dev` 容器内准备 PostgreSQL 表、Go 引擎和 Python 绑定，并设置 `XIAQI_DATABASE_URL`。MiniZero 现场采集需要 GPU，联调配置中的 Learner 使用 CPU。[数据库和引擎](#自对弈棋谱入库) · [Python 绑定](src/data/check_minizero/README.md#构建与运行)

```bash
export PYTHONPATH=/workspace/build/minizero/go
python -m scripts.train --config configs/training/go_9x9_smoke.toml
```

联调配置在 `smoke-go-9x9` 池采集两轮，每轮计划提交两盘并更新两步。命令会打印运行目录，目录内有 `config.toml`、`events.jsonl`、TensorBoard 数据、检查点和导出的采集模型。[训练控制器](src/control/training_controller/README.md) · [TensorBoard](#tensorboard)

续训时把 `--config` 指向原运行的 `config.toml`，用 `--resume` 指定 `checkpoints/` 下的检查点。续训沿用配置指定的棋谱池，在新运行目录中从恢复的训练步继续。

## 检查环境

```powershell
docker compose config --quiet
docker compose run --rm -T dev python scripts/check_env.py
```

检查会执行 GPU 张量运算，并使用 Inductor 后端实际编译 `torch.compile`
的前向和反向计算，与 eager 结果比较。首次运行需要等待编译，失败时返回非零退出码。

Inductor、Triton 和 CUDA 编译缓存均写入容器的 `/tmp`；
使用上述 `--rm` 命令退出后随容器清理。
