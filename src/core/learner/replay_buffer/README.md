# 内存经验池

`ReplayBuffer(board_size, min_train_samples, target_samples)` 在 CPU 内存中按盘保存复盘结果。`Learner` 调用它完成入池和采样，外层训练入口负责棋局的读取顺序、规则校验和去重。 [实现代码](replay_buffer.py) · [复盘样本](../../../data/minizero_replay/README.md#输出) · [回填和增量复盘](../../../../PostgreSQL.md#内存窗口与采样)

## 入池与淘汰

`add_samples(samples, position="newest")` 接收一盘的 `TrainingSamples`。增量棋局按入库先后加入最新一端。启动回填先读取最新棋局，随后使用 `position="oldest"` 加入更早的棋局。

三个数组须为 `float32` 且元素有限，行数相同。`features`、`policy`、`value` 的形状依次为 `(N, 18, n, n)`、`(N, n² + 1)` 和 `(N, 1)`。每行策略非负，概率和与 `1` 的绝对误差最多为 `1e-5`。价值只取 `-1`、`0` 或 `1`。校验成功后直接保存数组引用，调用方应将数组视为只读。有效的空批次直接返回当前池状态。

只要删除最老一盘后仍能保留至少 `target_samples` 条训练位置，池就淘汰该盘。每盘样本数可能不同，池会保留达到目标所需的完整棋局。`pool_status()` 和 `add_samples()` 均返回 `PoolStatus`。 [池状态字段](../../../data_types/pool_status/README.md)

## 抽样

`sample_batch(batch_size, rng)` 在池就绪后，从全部有效训练位置中等概率、无放回抽样。池更新后，下次抽样根据各盘的累计样本数定位盘内行号，再用同一组行号复制三组数组。`Learner` 收到批次后再应用棋盘旋转。 [棋盘旋转](../rotation/README.md)

## 测试

从仓库根目录运行。合成数据用例覆盖历史回填、增量更新、整盘淘汰、样本校验和三个数组的抽样对齐。 [测试代码](replay_buffer_test.py)

```sh
python -m unittest -v src.core.learner.replay_buffer.replay_buffer_test
```
