import math
import unittest
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from ...data_types.learner_config.learner_config import LearnerConfig
from ...data_types.pool_status.pool_status import PoolStatus
from ...data_types.step_metrics.step_metrics import StepMetrics
from ...data_types.training_samples.training_samples import TrainingSamples
from .learner import Learner


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = nn.Linear(18, 82)
        self.value = nn.Linear(18, 1)

    def forward(self, board: Tensor) -> tuple[Tensor, Tensor]:
        features = board.mean(dim=(2, 3))
        return self.policy(features), self.value(features).tanh()


class LearnerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = LearnerConfig(
            batch_size=3,
            min_train_samples=3,
            target_samples=10,
            seed=10,
            value_loss_weight=0.25,
        )
        rng = np.random.default_rng(0)
        features = rng.standard_normal((3, 18, 9, 9), dtype=np.float32)
        policy = np.zeros((3, 82), dtype=np.float32)
        policy[:, [0, 81]] = [[0.25, 0.75], [0, 1], [0.7, 0.3]]
        value = np.array([[-1], [0], [1]], dtype=np.float32)
        self.samples = TrainingSamples(features, policy, value)
        self.make_optimizer = partial(torch.optim.AdamW, lr=0.01, weight_decay=0.02)

    def test_updates_once_after_readiness_threshold(self) -> None:
        model = TinyModel().eval()
        make_model = Mock(return_value=model)
        learner = Learner(make_model, self.make_optimizer, self.config)
        make_model.assert_called_once_with()
        self.assertIs(learner.get_inference_model(), model)

        features, policy, value = (
            self.samples.features,
            self.samples.policy,
            self.samples.value,
        )
        learner.add_samples(TrainingSamples(features[:2], policy[:2], value[:2]))
        before = deepcopy(model.state_dict())
        self.assertEqual(learner.pool_status(), PoolStatus(2, 1, False))
        self.assertIsNone(learner.step())
        self.assertFalse(model.training)
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

        status = learner.add_samples(
            TrainingSamples(features[2:], policy[2:], value[2:])
        )
        self.assertEqual(status, PoolStatus(3, 2, True))
        rng = np.random.default_rng(self.config.seed)
        rows = rng.choice(3, size=3, replace=False)
        turns = rng.integers(0, 4, size=3)
        expected_board = torch.from_numpy(
            np.stack(
                [
                    np.rot90(features[row], turn, axes=(-2, -1))
                    for row, turn in zip(rows, turns)
                ]
            )
        )
        expected_policy = policy[rows].copy()
        for row, turn in enumerate(turns):
            board_policy = expected_policy[row, :-1].reshape(9, 9)
            expected_policy[row, :-1] = np.rot90(board_policy, turn).ravel()
        with torch.no_grad():
            logits, prediction = model(expected_board)
            policy_reference = cross_entropy(
                logits, torch.from_numpy(expected_policy)
            ).item()
            value_reference = mse_loss(prediction, torch.from_numpy(value[rows])).item()

        with patch.object(model, "forward", wraps=model.forward) as forward:
            metrics = learner.step()
        torch.testing.assert_close(
            forward.call_args.args[0], expected_board, rtol=0, atol=0
        )
        self.assertIsInstance(metrics, StepMetrics)
        self.assertEqual(metrics.step, 1)
        self.assertEqual(metrics.batch_size, 3)
        self.assertEqual(metrics.replay_samples, 3)
        self.assertAlmostEqual(metrics.policy_loss, policy_reference, places=5)
        self.assertAlmostEqual(metrics.value_loss, value_reference, places=6)
        self.assertAlmostEqual(
            metrics.total_loss, policy_reference + 0.25 * value_reference, places=5
        )
        for loss in (metrics.policy_loss, metrics.value_loss, metrics.total_loss):
            self.assertIsInstance(loss, float)
            self.assertTrue(math.isfinite(loss))
        self.assertTrue(model.training)
        self.assertIs(learner.get_inference_model(), model)
        for name in ("policy.weight", "value.weight"):
            self.assertFalse(torch.equal(model.state_dict()[name], before[name]))
        make_model.assert_called_once_with()

    def test_rejects_value_output_that_would_broadcast(self) -> None:
        class SqueezedValue(TinyModel):
            def forward(self, board: Tensor) -> tuple[Tensor, Tensor]:
                logits, value = super().forward(board)
                return logits, value.squeeze(-1)

        learner = Learner(SqueezedValue, self.make_optimizer, self.config)
        before = deepcopy(learner.get_inference_model().state_dict())
        learner.add_samples(self.samples)
        with self.assertRaisesRegex(ValueError, "value prediction shape"):
            learner.step()
        torch.testing.assert_close(learner.get_inference_model().state_dict(), before)

    def test_rejects_invalid_configuration(self) -> None:
        cases = (
            ({"board_size": 0}, "board_size"),
            ({"batch_size": 2.5}, "batch_size"),
            ({"batch_size": 4}, "batch_size"),
            ({"target_samples": 2}, "target_samples"),
            ({"value_loss_weight": -1}, "value_loss_weight"),
            ({"value_loss_weight": float("nan")}, "value_loss_weight"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    replace(self.config, **changes)

    def test_checkpoint_restores_weights_optimizer_and_step(self) -> None:
        learner = Learner(TinyModel, self.make_optimizer, self.config)
        learner.add_samples(self.samples)
        self.assertEqual(learner.step().step, 1)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "learner.pt"
            learner.save_checkpoint(path)
            original = torch.load(path, weights_only=True)

            resumed = Learner(
                TinyModel, partial(torch.optim.AdamW, lr=0.5), self.config
            )
            model = resumed.get_inference_model()
            resumed.load_checkpoint(str(path))
            self.assertIs(resumed.get_inference_model(), model)
            self.assertEqual(resumed.pool_status(), PoolStatus(0, 0, False))
            self.assertIsNone(resumed.step())
            resumed.save_checkpoint(path)
            restored = torch.load(path, weights_only=True)

        self.assertEqual(set(original), {"model", "optimizer", "step"})
        self.assertTrue(original["optimizer"]["state"])
        self.assertEqual(restored["step"], 1)
        torch.testing.assert_close(restored["model"], original["model"], rtol=0, atol=0)
        self.assertEqual(
            restored["optimizer"]["param_groups"], original["optimizer"]["param_groups"]
        )
        torch.testing.assert_close(
            restored["optimizer"]["state"],
            original["optimizer"]["state"],
            rtol=0,
            atol=0,
        )

        resumed.add_samples(self.samples)
        self.assertEqual(resumed.step().step, 2)
        self.assertFalse(
            torch.equal(model.policy.weight, original["model"]["policy.weight"])
        )

    def test_failed_checkpoint_write_preserves_previous_file(self) -> None:
        learner = Learner(TinyModel, self.make_optimizer, self.config)

        def fail_after_write(state, file) -> None:
            file.write(b"incomplete checkpoint")
            raise OSError("simulated write failure")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "learner.pt"
            learner.save_checkpoint(path)
            previous = path.read_bytes()
            with patch.object(torch, "save", side_effect=fail_after_write):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    learner.save_checkpoint(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_incompatible_optimizer_checkpoint_preserves_training_state(self) -> None:
        learner = Learner(TinyModel, self.make_optimizer, self.config)
        learner.add_samples(self.samples)
        self.assertEqual(learner.step().step, 1)
        model = learner.get_inference_model()
        before = deepcopy(model.state_dict())

        with TemporaryDirectory() as directory:
            path = Path(directory) / "learner.pt"
            learner.save_checkpoint(path)
            invalid = torch.load(path, weights_only=True)
            invalid["model"] = TinyModel().state_dict()
            invalid["optimizer"]["param_groups"] = []
            invalid["step"] = 10
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "parameter groups"):
                learner.load_checkpoint(path)

        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        self.assertEqual(learner.step().step, 2)

    def test_compiled_step_updates_raw_model_and_checkpoint_loads_in_eager(
        self,
    ) -> None:
        model = TinyModel()
        before = deepcopy(model.state_dict())
        config = replace(self.config, compile_model=True)
        # Exercise torch.compile and compiled autograd without a native compiler.
        compile_for_test = partial(torch.compile, backend="aot_eager")
        with patch.object(
            torch, "compile", side_effect=compile_for_test
        ) as compile_model:
            learner = Learner(lambda: model, self.make_optimizer, config)
        compile_model.assert_called_once_with(model)

        learner.add_samples(self.samples)
        self.assertEqual(learner.step().step, 1)
        self.assertIs(learner.get_inference_model(), model)
        self.assertFalse(torch.equal(model.policy.weight, before["policy.weight"]))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "learner.pt"
            learner.save_checkpoint(path)
            resumed = Learner(TinyModel, self.make_optimizer, self.config)
            resumed.load_checkpoint(path)

        torch.testing.assert_close(
            resumed.get_inference_model().state_dict(), model.state_dict()
        )
        resumed.add_samples(self.samples)
        self.assertEqual(resumed.step().step, 2)


if __name__ == "__main__":
    unittest.main()
