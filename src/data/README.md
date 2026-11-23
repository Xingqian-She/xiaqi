# 数据模块

采集模块取得完整棋谱并入库，训练控制器按池读取和复盘，生成 Learner 所用的样本。 [棋谱数据指南](../../PostgreSQL.md)

| 模块 | 职责 |
| --- | --- |
| [MiniZero 自对弈](minizero_selfplay/README.md) | 管理引擎进程、导出生效配置并筛选完整对局消息。 |
| [PostgreSQL 存储](postgresql_store/README.md) | 登记采集来源，按池分配入库编号并逐盘提交。 |
| [PostgreSQL 读取](postgresql_reader/README.md) | 按池和入库编号分页读取棋谱与来源信息。 |
| [棋谱解析](minizero_record/README.md) | 从棋谱文本读取落子、搜索权重、终局结果和样本范围。 |
| [围棋复盘](minizero_replay/README.md) | 用 MiniZero 环境校验整局，生成特征、策略和价值数组。 |
| [Python 环境检查](check_minizero/README.md) | 构建和加载 Go 版 `minizero_py`，用固定短局验证绑定行为。 |
