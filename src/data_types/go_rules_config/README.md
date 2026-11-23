# GoRulesConfig

围棋规则由 `board_size`、`komi` 和 `ko_rule` 组成，分别映射到 MiniZero 的 `env_board_size`、`env_go_komi` 和 `env_go_ko_rule`。默认使用 `9` 路、`7.0` 目和 `positional`，劫规则也可选 `situational`。训练配置用同一棋盘尺寸构造模型和经验池。 [类型定义](go_rules_config.py) · [训练配置](../../control/config_io/README.md)
