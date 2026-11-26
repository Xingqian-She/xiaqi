# 数据模块

采集模块将完整棋谱写入所选实验的库，训练控制器读取并复盘，生成 Learner 所用的样本。 [棋谱数据指南](../../PostgreSQL.md)

| 模块 | 职责 |
| --- | --- |
| [MiniZero 自对弈](minizero_selfplay/README.md) | 管理引擎进程、导出生效配置并筛选完整对局消息。 |
| [PostgreSQL 棋谱库](postgresql/README.md) | 初始化共用结构、登记采集来源、逐盘提交并按入库编号分页。 |
| [棋谱解析](minizero_record/README.md) | 从棋谱文本读取落子、搜索权重、终局结果和样本范围。 |
| [围棋复盘](minizero_replay/README.md) | 用 MiniZero 环境校验整局，生成特征、策略和价值数组。 |
| [Python 环境检查](check_minizero/README.md) | 构建和加载 Go 版 `minizero_py`，用固定短局验证绑定行为。 |
