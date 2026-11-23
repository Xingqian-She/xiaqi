# 自对弈采集和导入

`collect_live(store, *, pool_id, games, engine_path, config_path, model_path, overrides=...)` 启动 MiniZero，提交指定数目的完整棋谱。控制器用 `training_overrides(config)` 生成类型化覆盖，采集器在启动前核对 TorchScript 元数据和引擎生效参数。独立采集 CLI 把 `--conf-str` 原样传入覆盖参数，最后设置本批的权重路径。 [实现代码](selfplay_collector.py) · [采集 CLI](../../../PostgreSQL.md#采集与导入)

控制器使用标准 MCTS。`training_overrides()` 固定 `actor_use_gumbel=false`、`actor_use_gumbel_noise=false` 和 `zero_actor_intermediate_sequence_length=0`。棋谱的 `P` 记录访问次数，复盘时归一化为策略目标。 [搜索参数映射](../../data_types/alphazero_config/README.md#minizero-参数映射) · [复盘输出](../../data/minizero_replay/README.md#输出)

`import_records(store, source, record_file)` 从已有输出文件导入完整棋谱。同一来源 UUID 和相同盘序可以重试，返回已确认的原入库编号。两个入口都返回 `SelfPlayResult`，并接受来源登记后和逐盘提交后的回调。[采集结果](../../data_types/selfplay_result/README.md) · [存储事务](../../data/postgresql_store/README.md#事务与重试)

现场采集盘数不足、导入文件没有完整棋谱、处理失败或入库冲突时抛出异常。之前已经提交的棋谱可供重新读取，现场采集退出时会收尾子进程。[进程管理](../../data/minizero_selfplay/README.md#现场自对弈)

测试覆盖导入重试的 ID 稳定性、TorchScript 元数据和引擎生效配置。 [测试代码](selfplay_collector_test.py)
