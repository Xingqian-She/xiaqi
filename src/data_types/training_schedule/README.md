# TrainingSchedule

`games_per_round`、`train_steps_per_round` 和 `max_rounds` 分别指定每轮采集的盘数、达到门槛后执行的更新步数和本次启动的采集轮数，均为正整数。`warmup_train_steps` 和 `min_new_games_before_train` 默认为 `0`。盘数门槛按最近一次常规训练后本控制器采集且复盘产出样本的棋局计数。 [类型定义](training_schedule.py) · [控制器](../../control/training_controller/README.md)
