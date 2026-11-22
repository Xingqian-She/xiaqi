"""围棋自对弈样本的内存经验池。"""

from collections import deque
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ....data.minizero_replay.training_samples.training_samples import TrainingSamples
from .pool_status.pool_status import PoolStatus


class ReplayBuffer:
    """按入库顺序保存整盘样本，按训练位置均匀抽样。"""

    def __init__(
        self, board_size: int, min_train_samples: int, target_samples: int
    ) -> None:
        for name, size in (
            ("board_size", board_size),
            ("min_train_samples", min_train_samples),
            ("target_samples", target_samples),
        ):
            if not isinstance(size, Integral) or isinstance(size, bool) or size < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if min_train_samples > target_samples:
            raise ValueError("min_train_samples must not exceed target_samples.")

        board_size = int(board_size)
        self._feature_shape = (18, board_size, board_size)
        self._policy_size = board_size**2 + 1
        self._min_train_samples = int(min_train_samples)
        self._target_samples = int(target_samples)
        self._games: deque[TrainingSamples] = deque()
        self._sample_count = 0
        self._game_ends: NDArray[np.int64] | None = None

    def pool_status(self) -> PoolStatus:
        return PoolStatus(
            sample_count=self._sample_count,
            game_count=len(self._games),
            ready=self._sample_count >= self._min_train_samples,
        )

    def add_samples(
        self,
        samples: TrainingSamples,
        *,
        position: Literal["oldest", "newest"] = "newest",
    ) -> PoolStatus:
        """加入一盘样本。入池后调用方应将数组视为只读。"""
        if position not in ("oldest", "newest"):
            raise ValueError("position must be 'oldest' or 'newest'.")
        sample_count = self._validate_samples(samples)
        if sample_count == 0:
            return self.pool_status()

        if position == "oldest":
            self._games.appendleft(samples)
        else:
            self._games.append(samples)
        self._sample_count += sample_count

        while self._games:
            oldest_count = self._games[0].features.shape[0]
            if self._sample_count - oldest_count < self._target_samples:
                break
            self._games.popleft()
            self._sample_count -= oldest_count

        self._game_ends = None
        return self.pool_status()

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> TrainingSamples:
        """组装一个批内不重复的 CPU 样本批次。"""
        if self._sample_count < self._min_train_samples:
            raise RuntimeError(
                f"Replay buffer has {self._sample_count} samples. "
                f"Training requires at least {self._min_train_samples}."
            )
        if (
            not isinstance(batch_size, Integral)
            or isinstance(batch_size, bool)
            or not 1 <= batch_size <= self._sample_count
        ):
            raise ValueError(
                "batch_size must be an integer between 1 and sample_count."
            )

        if self._game_ends is None:
            self._game_ends = np.cumsum(
                [game.features.shape[0] for game in self._games], dtype=np.int64
            )
        game_ends = self._game_ends
        indices = rng.choice(self._sample_count, size=batch_size, replace=False)
        game_indices = game_ends.searchsorted(indices, side="right")

        features = np.empty((batch_size, *self._feature_shape), dtype=np.float32)
        policy = np.empty((batch_size, self._policy_size), dtype=np.float32)
        value = np.empty((batch_size, 1), dtype=np.float32)
        for row, (index, game_index) in enumerate(zip(indices, game_indices)):
            game = self._games[game_index]
            game_start = game_ends[game_index - 1] if game_index else 0
            move = index - game_start
            features[row] = game.features[move]
            policy[row] = game.policy[move]
            value[row] = game.value[move]
        return TrainingSamples(features=features, policy=policy, value=value)

    def _validate_samples(self, samples: TrainingSamples) -> int:
        sample_count = None
        for name, array, row_shape in (
            ("features", samples.features, self._feature_shape),
            ("policy", samples.policy, (self._policy_size,)),
            ("value", samples.value, (1,)),
        ):
            if not isinstance(array, np.ndarray) or array.dtype != np.float32:
                raise ValueError(f"{name} must be a float32 NumPy array.")
            if array.shape[1:] != row_shape:
                shape = ", ".join(str(size) for size in row_shape)
                raise ValueError(f"{name} must have shape (N, {shape}).")
            if sample_count is None:
                sample_count = array.shape[0]
            elif array.shape[0] != sample_count:
                raise ValueError(f"{name} must have {sample_count} rows.")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values.")

        if (samples.policy < 0).any() or not np.allclose(
            samples.policy.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-5
        ):
            raise ValueError("Each policy row must be nonnegative and sum to 1.")
        if not np.isin(samples.value, (-1, 0, 1)).all():
            raise ValueError("value must contain only -1, 0 or 1.")
        return samples.features.shape[0]
