# xiaqi

首版实现 9×9 围棋 AlphaZero 自对弈训练。MiniZero 使用策略与价值网络引导搜索，PostgreSQL 保存棋谱，PyTorch Transformer 从复盘样本中学习。每次完成训练阶段后，控制器导出模型供下一轮采集使用。

旧版代码和资料保留在 `Zero-TTT` 分支。

## 开发环境

需要 Git、支持 `gpus` 的 Docker Compose，以及可供 Linux 容器使用的 NVIDIA GPU。Windows 使用 Docker Desktop 的 WSL 2 后端。 [环境检查与构建代理](docs/training.md#检查开发环境) · [Compose 配置](compose.yaml)

## 联调快跑

首次检出后，在仓库根目录用 PowerShell 初始化子模块、构建开发镜像与 MiniZero，再启动联调。

```powershell
git submodule update --init -- third_party/minizero
docker compose build dev
docker compose run --rm -T dev bash scripts/build_minizero.sh
docker compose run --rm -T dev bash scripts/build_minizero.sh --python
.\scripts\run.ps1 smoke -New
```

联调从新建的模型开始。Learner 使用 CPU，MiniZero 自对弈使用 GPU。训练配置、检查点和独立棋谱库都在 `runs/smoke/`。下次用 `.\scripts\run.ps1 smoke -Resume` 在原目录续训。 [联调结果与运行产物](docs/training.md#运行产物与进度)

## 文档索引

| 要查的内容 | 文档 |
| --- | --- |
| 选择训练配置、看日志、从检查点续训 | [自对弈训练指南](docs/training.md) |
| 建表、采集和导入棋谱、回填经验池 | [PostgreSQL 棋谱数据指南](PostgreSQL.md) |
| 构建引擎、获取公开权重、调试 GTP | [MiniZero 构建与运行](third_party/README.md#minizero) |
| 训练流程、配置读写、采集调度与日志 | [训练控制模块](src/control/README.md) |
| 环境绑定、棋谱解析、复盘和数据库接口 | [数据模块](src/data/README.md) |
| 模型结构、参数更新、经验池与检查点 | [核心组件](src/core/README.md) |
| 配置字段与数据结构 | [数据类](src/data_types/README.md) |
| TorchScript 推理接口与模型导出 | [MiniZero 模型适配](src/integrations/minizero/README.md) |
