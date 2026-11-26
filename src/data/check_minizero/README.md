# MiniZero Python 环境检查

用项目的 9×9 Go 配置加载 `minizero_py`，从 Python 执行固定短局，核对复盘会用到的环境行为。检查只用 CPU，无需模型权重。 [检查脚本](check_minizero.py) · [合并后的 Python 绑定](https://github.com/rlglab/minizero/pull/13)

## 构建与运行

已有项目的 `dev` 镜像和 MiniZero 子模块时，从仓库根目录运行。`--python` 将 Go 版 `minizero_py*.so` 构建到 `build/minizero/go/`，使用同一个镜像的 Python 加载它。 [开发环境](../../../README.md#开发环境) · [构建脚本](../../../scripts/build_minizero.sh)

```powershell
docker compose run --rm --no-deps --pull never -T dev bash scripts/build_minizero.sh --python
docker compose run --rm --no-deps --pull never -T -e PYTHONPATH=/workspace/build/minizero/go dev `
  python -m src.data.check_minizero.check_minizero
```

脚本打印扩展位置、配置和输入形状，然后执行黑 `B1`、白 `A1`、黑 `A2` 提白一子、白停、黑停。期间还会尝试一次落在已占用的 `B1`，检查拒绝落子时局面保持一致。预期历史为 `[(1, 1), (0, 2), (9, 1), (81, 2), (81, 1)]`，终局 `eval_score=+1`。复位后恢复空盘和黑先，成功时输出 `MiniZero Go environment check passed.`，检查失败时返回非零退出码。

## 复盘会用到的约定

| 接口 | 当前 Go 绑定的行为 |
| --- | --- |
| `load_config_file(path)`、`Environment()` | 先加载配置并检查返回值，再构造环境。配置作用于进程，复盘过程中应使用固定配置。 |
| `policy_size()`、`legal_actions()` | 9×9 策略长度为 `82`。棋盘坐标从零开始，`action_id = row * 9 + col`，`row=0` 对应 GTP 第 1 行，列字母跳过 `I`。`pass` 的 ID 是 `81`。 |
| `turn()`、`act(action_id)`、`action_history()` | 黑为 `1`，白为 `2`。`act` 使用当前执子方，历史元素是 `(action_id, player)`，包含 `pass`。复盘时用 `turn()` 核对棋谱记录的执子方。 |
| `is_legal_action(action_id)`、`is_terminal()` | 复盘时先检查终局和 ID 范围，再查合法性。已占用的点会被拒绝。当前 Go 环境判为终局后仍可报告并执行合法动作，脚本会打印终局后的 `pass` 查询结果。 |
| `features(rotation=0)` | 返回 `float32` 的 NumPy 快照，形状为 `(18, 9, 9)`。通道 `0–15` 每两层为一组，从最新局面起保存最近 8 个局面，按当前执子方的己方、对方排列。通道 `16` 和 `17` 分别表示当前是否轮到黑、白。 |
| `action_features(action_id, rotation=0)` | 返回 `(1, 9, 9)` 的 `float32` 平面。落子点为 `1`，其余为 `0`，`pass` 全为 `0`。两种特征的旋转编号均为 `0–7`，`0` 保持原方向。 |
| `reward()`、`eval_score()` | Go 的逐手奖励为 `0`。`eval_score()` 按黑方视角返回 `+1` 黑胜、`-1` 白胜、`0` 和棋。若生成当前执子方视角的价值标签，白方样本将结果取反。 |

动作 ID 须在 `0 <= action_id < policy_size()` 内。传入越界 ID、无效的玩家编号或旋转编号时，底层 C++ 可能触发断言或产生不安全的访问。显式玩家重载 `is_legal_action(action_id, player)` 用指定颜色检查规则，棋谱是否轮到该颜色由调用方核对。 [绑定源码](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/learner/pybind.cpp) · [Go 环境源码](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/environment/go/go.cpp) · [坐标转换](https://github.com/rlglab/minizero/blob/3db2d41901057c7da5726cd961d070f01197636f/minizero/utils/sgf_loader.cpp)

## 测试

```powershell
docker compose run --rm --no-deps --pull never -T -e PYTHONPATH=/workspace/build/minizero/go dev `
  python -m unittest -v src.data.check_minizero.check_minizero_test
```

测试覆盖导入和配置失败、完整短局，以及项目的 `7` 目贴目是否参与终局判定。找不到 `minizero_py` 时，原生绑定用例会跳过。 [测试代码](check_minizero_test.py) · [9×9 配置](../../../runs/go-9x9-az/minizero_base.cfg)
