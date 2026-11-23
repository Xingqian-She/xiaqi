# 第三方依赖

## MiniZero

`third_party/minizero/` 是 MiniZero 的 Git 子模块，固定到 `3db2d41`。这个版本包含 Python 环境绑定和 Apache 2.0 许可证。本项目在 `dev` 容器中构建围棋引擎 `minizero_go`，并链接该容器的 PyTorch C++ 库。 [上游提交](https://github.com/rlglab/minizero/commit/3db2d41901057c7da5726cd961d070f01197636f) · [许可证](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/LICENSE)

### 构建

按首页初始化子模块、创建数据卷和构建 `dev` 镜像，再从仓库根目录用 PowerShell 构建 MiniZero。训练需要引擎和 Go 版 Python 绑定，下面的 GTP 和独立自对弈调试使用引擎。 [首次构建](../README.md#联调快跑) · [构建脚本](../scripts/build_minizero.sh)

```powershell
docker compose run --rm --no-deps -T dev bash scripts/build_minizero.sh
docker compose run --rm --no-deps -T dev bash scripts/build_minizero.sh --python
```

构建产物在 `build/minizero/go/`。引擎构建生成 `minizero_go` 和采集时读取的 `minizero_go.commit`，`--python` 生成 `minizero_py*.so`。 [Python 绑定检查](../src/data/check_minizero/README.md#构建与运行)

构建脚本在容器的临时源码副本上应用 `-mode sp` 命令补丁，修复 `start` 读取队列和 `quit` 退出时可能卡住的问题。 [构建补丁](minizero-sp-commands.patch)

### 公开权重

GTP 和下面的自对弈调试示例使用 MiniZero 公开的 9×9 Go AlphaZero 权重。从下载页的 9x9 Go 的 Trained Models 获取权重包，解压出 `n=200` 的 AlphaZero 权重，将其保存为 `models/minizero/go_9x9_az_n200.pt`。这两个示例使用项目的 9×9 配置。 [MiniZero 公开权重下载页](https://rlg.iis.sinica.edu.tw/papers/minizero/#downloads) · [9×9 配置](../configs/minizero/go_9x9_az.cfg)

### GTP 落子

```powershell
docker compose run --rm --no-deps dev ./build/minizero/go/minizero_go `
  -mode console -conf_file configs/minizero/go_9x9_az.cfg `
  -conf_str 'nn_file_name=models/minizero/go_9x9_az_n200.pt:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_use_dirichlet_noise=false'
```

进入控制台后输入 `genmove B`，引擎会返回黑方的着法。可输入 `showboard` 查看局面，输入 `quit` 退出。

### 自对弈

```powershell
docker compose run --rm --no-deps dev ./build/minizero/go/minizero_go `
  -mode sp -conf_file configs/minizero/go_9x9_az.cfg `
  -conf_str 'nn_file_name=models/minizero/go_9x9_az_n200.pt:zero_num_parallel_games=1:zero_num_threads=1:zero_disable_resign_ratio=1'
```

输入 `start` 开始。等标准输出出现 `SelfPlay true … #` 后，输入 `quit` 退出。每条完整记录包含走子、各手搜索访问次数 `P` 和黑方视角的终局结果 `RE`。
