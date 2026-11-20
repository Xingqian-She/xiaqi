# 第三方依赖

## MiniZero

`third_party/minizero/` 是 MiniZero 的 Git 子模块，固定到 `3db2d41`。这个版本包含 Python 环境绑定和 Apache 2.0 许可证。本项目在 `dev` 容器中构建围棋引擎 `minizero_go`，并链接该容器的 PyTorch C++ 库。 [上游提交](https://github.com/rlglab/minizero/commit/3db2d41901057c7da5726cd961d070f01197636f) · [许可证](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/LICENSE)

### 构建

在仓库根目录使用 PowerShell。首次检出项目时初始化子模块。网络连接超时可使用本仓库的代理配置构建镜像。 [代理配置](../README.md#开发环境)

```powershell
git submodule update --init -- third_party/minizero
docker compose build dev
docker compose run --rm --no-deps -T dev bash scripts/build_minizero.sh
```

构建产物在 `build/minizero/go/minizero_go`。运行时使用 `configs/minizero/go_9x9_az.cfg` 和本地模型 `models/minizero/go_9x9_az_n200.pt`。 [MiniZero 公开权重及配置](https://rlg.iis.sinica.edu.tw/papers/minizero/#downloads)

构建脚本在容器的临时源码副本上应用 `-mode sp` 命令补丁，修复 `start` 读取队列和 `quit` 退出时可能卡住的问题。 [构建补丁](minizero-sp-commands.patch)

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
