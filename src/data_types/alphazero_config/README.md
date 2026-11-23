# AlphaZeroConfig

`[alphazero]` 配置 MiniZero MCTS 的搜索预算、PUCT、价值回传、探索噪声、特征旋转和认输。 [类型定义](alphazero_config.py) · [采集覆盖参数](../../control/selfplay_collector/selfplay_collector.py)

## MiniZero 参数映射

| TOML 字段 | MiniZero 配置项 | 默认值 |
| --- | --- | --- |
| `num_simulations` | `actor_num_simulation` | `200` |
| `puct_base`、`puct_init` | `actor_mcts_puct_base`、`actor_mcts_puct_init` | `19652.0`、`1.25` |
| `reward_discount`、`value_rescale` | `actor_mcts_reward_discount`、`actor_mcts_value_rescale` | `1.0`、`false` |
| `action_selection` | `actor_select_action_by_count`、`actor_select_action_by_softmax_count` | `"softmax_count"` |
| `temperature` | `actor_select_action_softmax_temperature` | `1.0` |
| `use_dirichlet_noise` | `actor_use_dirichlet_noise` | `true` |
| `dirichlet_alpha`、`dirichlet_epsilon` | `actor_dirichlet_noise_alpha`、`actor_dirichlet_noise_epsilon` | `0.11`、`0.25` |
| `random_rotation_features` | `actor_use_random_rotation_features` | `true` |
| `resign_threshold`、`disable_resign_ratio` | `actor_resign_threshold`、`zero_disable_resign_ratio` | `-0.9`、`0.1` |

`max_count` 选择访问次数最多的动作，`softmax_count` 以给定温度按访问次数采样。配置值只开启对应的一个落子开关。`disable_resign_ratio` 为 `1.0` 时整批禁用认输。 [训练配置对照](../../../docs/training.md#选择训练配置)
