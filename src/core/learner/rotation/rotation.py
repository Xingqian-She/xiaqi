"""按样本随机旋转棋盘特征和策略落点。"""

import numpy as np

from ....data_types.training_samples.training_samples import TrainingSamples


def random_rotate_samples(
    samples: TrainingSamples, rng: np.random.Generator
) -> TrainingSamples:
    """每条样本独立等概率选择一个直角旋转，返回新的批次。"""
    batch_size, channels, board_size, _ = samples.features.shape
    num_points = board_size**2
    grid = np.arange(num_points).reshape(board_size, board_size)
    # 每张索引表给出旋转后各落点对应的原位置。
    rotation_indices = np.stack([np.rot90(grid, turns).ravel() for turns in range(4)])
    turns = rng.integers(0, 4, size=batch_size)
    indices = rotation_indices[turns]

    features = np.take_along_axis(
        samples.features.reshape(batch_size, channels, num_points),
        indices[:, None, :],
        axis=2,
    ).reshape(samples.features.shape)
    policy = samples.policy.copy()
    policy[:, :-1] = np.take_along_axis(samples.policy[:, :-1], indices, axis=1)
    return TrainingSamples(features, policy, samples.value.copy())
