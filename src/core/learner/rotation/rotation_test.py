import unittest
from unittest.mock import Mock

import numpy as np

from ....data_types.training_samples.training_samples import TrainingSamples
from .rotation import random_rotate_samples


class RotationTest(unittest.TestCase):
    def test_rotates_all_feature_planes_and_policy_with_same_angle(self) -> None:
        points = np.arange(9, dtype=np.float32)
        offsets = 10 * np.arange(4 * 18, dtype=np.float32).reshape(4, 18, 1)
        features = (offsets + points).reshape(4, 18, 3, 3)
        weights = np.arange(1, 41, dtype=np.float32).reshape(4, 10)
        policy = weights / weights.sum(axis=1, keepdims=True)
        value = np.array([[-1], [0], [1], [-1]], dtype=np.float32)
        for array in (features, policy, value):
            array.setflags(write=False)
        samples = TrainingSamples(features, policy, value)
        rng = Mock(spec=np.random.Generator)
        rng.integers.return_value = np.arange(4)

        rotated = random_rotate_samples(samples, rng)

        rng.integers.assert_called_once_with(0, 4, size=4)
        positions = (
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 5, 8, 1, 4, 7, 0, 3, 6),
            (8, 7, 6, 5, 4, 3, 2, 1, 0),
            (6, 3, 0, 7, 4, 1, 8, 5, 2),
        )
        for row, order in enumerate(positions):
            with self.subTest(quarter_turns=row):
                np.testing.assert_array_equal(
                    rotated.features[row].reshape(18, 9),
                    offsets[row] + points[list(order)],
                )
                np.testing.assert_array_equal(
                    rotated.policy[row, :-1], policy[row, list(order)]
                )

        np.testing.assert_array_equal(rotated.policy[:, -1], policy[:, -1])
        np.testing.assert_array_equal(rotated.value, value)
        for original, output in (
            (features, rotated.features),
            (policy, rotated.policy),
            (value, rotated.value),
        ):
            self.assertEqual(output.shape, original.shape)
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(output.flags.c_contiguous)
            self.assertFalse(np.shares_memory(original, output))

    def test_seed_reproduces_rotation_on_larger_board(self) -> None:
        features = np.arange(3 * 18 * 13**2, dtype=np.float32).reshape(3, 18, 13, 13)
        policy = np.eye(170, dtype=np.float32)[[11, 27, 30]]
        value = np.zeros((3, 1), dtype=np.float32)
        samples = TrainingSamples(features, policy, value)

        rotated = random_rotate_samples(samples, np.random.default_rng(0))
        repeated = random_rotate_samples(samples, np.random.default_rng(0))

        np.testing.assert_array_equal(rotated.features, repeated.features)
        np.testing.assert_array_equal(rotated.policy, repeated.policy)
        self.assertFalse(np.array_equal(rotated.features, features))

    def test_empty_batch_preserves_shapes(self) -> None:
        samples = TrainingSamples(
            np.empty((0, 18, 9, 9), dtype=np.float32),
            np.empty((0, 82), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )
        rotated = random_rotate_samples(samples, np.random.default_rng(0))
        self.assertEqual(rotated.features.shape, samples.features.shape)
        self.assertEqual(rotated.policy.shape, samples.policy.shape)
        self.assertEqual(rotated.value.shape, samples.value.shape)


if __name__ == "__main__":
    unittest.main()
