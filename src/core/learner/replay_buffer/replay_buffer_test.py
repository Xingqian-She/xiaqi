import unittest
from dataclasses import replace

import numpy as np

from ....data.minizero_replay.minizero_replay import TrainingSamples
from .replay_buffer import PoolStatus, ReplayBuffer


BOARD_SIZE = 3
POLICY_SIZE = BOARD_SIZE**2 + 1


def make_samples(start: int, count: int) -> TrainingSamples:
    ids = np.arange(start, start + count)
    features = np.empty((count, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features[:] = ids[:, None, None, None]
    policy = np.eye(POLICY_SIZE, dtype=np.float32)[ids % POLICY_SIZE]
    value = (ids % 3 - 1).astype(np.float32)[:, None]
    return TrainingSamples(features, policy, value)


class ReplayBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = ReplayBuffer(BOARD_SIZE, min_train_samples=4, target_samples=6)

    def assert_retains_rows(self, start: int, count: int) -> None:
        batch = self.buffer.sample_batch(count, np.random.default_rng(1))
        np.testing.assert_array_equal(
            np.sort(batch.features[:, 0, 0, 0]), np.arange(start, start + count)
        )

    def test_backfills_oldest_and_evicts_entire_games_on_refresh(self) -> None:
        self.assertEqual(
            self.buffer.add_samples(make_samples(7, 3), position="oldest"),
            PoolStatus(3, 1, False),
        )
        self.assertEqual(
            self.buffer.add_samples(make_samples(5, 2), position="oldest"),
            PoolStatus(5, 2, True),
        )
        self.assertEqual(
            self.buffer.add_samples(make_samples(1, 4), position="oldest"),
            PoolStatus(9, 3, True),
        )
        self.assert_retains_rows(1, 9)

        self.assertEqual(
            self.buffer.add_samples(make_samples(10, 2)), PoolStatus(7, 3, True)
        )
        self.assert_retains_rows(5, 7)
        self.assertEqual(
            self.buffer.add_samples(make_samples(12, 5)), PoolStatus(7, 2, True)
        )
        self.assert_retains_rows(10, 7)

        self.assertEqual(
            self.buffer.add_samples(make_samples(0, 1), position="oldest"),
            PoolStatus(7, 2, True),
        )
        self.assert_retains_rows(10, 7)

    def test_samples_global_rows_with_shared_indices(self) -> None:
        self.buffer.add_samples(make_samples(4, 1))
        older = make_samples(0, 4)
        older.features.setflags(write=False)
        self.buffer.add_samples(older, position="oldest")
        self.buffer.add_samples(make_samples(5, 1))

        seed = 21
        indices = np.random.default_rng(seed).choice(6, size=4, replace=False)
        expected = make_samples(0, 6)
        batch = self.buffer.sample_batch(4, np.random.default_rng(seed))
        for field in ("features", "policy", "value"):
            with self.subTest(field=field):
                array = getattr(batch, field)
                self.assertEqual(array.dtype, np.float32)
                np.testing.assert_array_equal(array, getattr(expected, field)[indices])
        self.assertEqual(len(np.unique(batch.features[:, 0, 0, 0])), 4)
        self.assertFalse(np.shares_memory(batch.features, older.features))

    def test_empty_game_does_not_change_pool(self) -> None:
        empty = make_samples(0, 0)
        self.assertEqual(self.buffer.add_samples(empty), PoolStatus(0, 0, False))
        status = self.buffer.add_samples(make_samples(1, 6))
        self.assertEqual(self.buffer.add_samples(empty, position="oldest"), status)
        self.assert_retains_rows(1, 6)

    def test_rejects_invalid_batches_without_evicting_games(self) -> None:
        status = self.buffer.add_samples(make_samples(1, 6))
        candidate = make_samples(7, 6)
        negative_policy = candidate.policy.copy()
        negative_policy[0, candidate.policy[0].argmax()] = 1.25
        negative_policy[0, 0] = -0.25
        cases = (
            (
                replace(candidate, features=candidate.features[:, :17]),
                "features.*shape",
            ),
            (
                replace(candidate, features=candidate.features.astype(np.float64)),
                "float32",
            ),
            (
                replace(candidate, features=np.full_like(candidate.features, np.nan)),
                "finite",
            ),
            (replace(candidate, policy=candidate.policy[:-1]), "policy.*6 rows"),
            (replace(candidate, policy=candidate.policy[:, :-1]), "policy.*shape"),
            (
                replace(candidate, policy=np.full_like(candidate.policy, np.inf)),
                "finite",
            ),
            (replace(candidate, policy=negative_policy), "nonnegative"),
            (replace(candidate, policy=candidate.policy * 0.5), "sum to 1"),
            (replace(candidate, value=candidate.value[:, 0]), "value.*shape"),
            (replace(candidate, value=np.full_like(candidate.value, np.nan)), "finite"),
            (
                replace(candidate, value=np.full_like(candidate.value, 0.5)),
                "-1, 0 or 1",
            ),
            (replace(make_samples(0, 0), value=candidate.value), "value.*0 rows"),
        )
        for samples, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.buffer.add_samples(samples)
                self.assertEqual(self.buffer.pool_status(), status)
        with self.assertRaisesRegex(ValueError, "position"):
            self.buffer.add_samples(candidate, position="middle")
        self.assertEqual(self.buffer.pool_status(), status)
        self.assert_retains_rows(1, 6)

    def test_sampling_requires_ready_pool_and_valid_batch_size(self) -> None:
        rng = np.random.default_rng(0)
        self.buffer.add_samples(make_samples(1, 3))
        with self.assertRaisesRegex(RuntimeError, "has 3 samples"):
            self.buffer.sample_batch(2, rng)

        self.buffer.add_samples(make_samples(4, 1))
        for batch_size in (0, 5, 2.5):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    self.buffer.sample_batch(batch_size, rng)


if __name__ == "__main__":
    unittest.main()
