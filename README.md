# xiaqi

重新开始的下棋 AI 项目，使用 Docker 提供 Python/PyTorch GPU 开发环境。
旧版代码和资料保留在 `Zero-TTT` 分支。

9×9 围棋推理和自对弈见 [MiniZero 构建与运行](third_party/README.md#minizero)。

## 开发环境

需要 Docker Compose，以及可供 Linux 容器使用的 NVIDIA GPU。
沿用基础镜像 `pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel`，
提供 Python、PyTorch、CUDA、cuDNN、Triton、C/C++ 编译工具和 MiniZero 的构建依赖。

在仓库根目录构建并进入容器：

```powershell
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

## 检查环境

```powershell
docker compose config --quiet
docker compose run --rm -T dev python scripts/check_env.py
```

检查会执行 GPU 张量运算，并使用 Inductor 后端实际编译 `torch.compile`
的前向和反向计算，与 eager 结果比较。首次运行需要等待编译，失败时返回非零退出码。

Inductor、Triton 和 CUDA 编译缓存均写入容器的 `/tmp`；
使用上述 `--rm` 命令退出后随容器清理。
