import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from ....data_types.training_samples.training_samples import TrainingSamples
from .gradient_accumulation import train_accumulated_batch


class SmallNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(2, 4)

    def forward(self, board: Tensor) -> tuple[Tensor, Tensor]:
        logits, value = self.output(board.flatten(1)).split((3, 1), dim=1)
        return logits, value


class GradientAccumulationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = SmallNetwork()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.5)
        self.samples = TrainingSamples(
            np.random.default_rng(0).standard_normal((6, 2, 1, 1), dtype=np.float32),
            np.eye(3, dtype=np.float32)[[1, 0, 2, 1, 2, 0]],
            np.array([[1], [1], [0], [-1], [0], [-1]], dtype=np.float32),
        )

    def accumulate(self) -> tuple[float, float, float]:
        return train_accumulated_batch(
            self.model,
            self.optimizer,
            self.samples,
            device=torch.device("cpu"),
            mini_batch_size=2,
            value_loss_weight=0.25,
        )

    def test_matches_full_batch_update_and_metrics(self) -> None:
        reference = deepcopy(self.model)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.03, momentum=0.5
        )
        logits, value = reference(torch.from_numpy(self.samples.features))
        policy_loss = cross_entropy(logits, torch.from_numpy(self.samples.policy))
        value_loss = mse_loss(value, torch.from_numpy(self.samples.value))
        total_loss = policy_loss + 0.25 * value_loss
        total_loss.backward()
        reference_optimizer.step()

        with (
            patch.object(self.model, "forward", wraps=self.model.forward) as forward,
            patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step,
            patch.object(
                self.optimizer, "zero_grad", wraps=self.optimizer.zero_grad
            ) as zero_grad,
        ):
            losses = self.accumulate()

        self.assertEqual(
            [call.args[0].size(0) for call in forward.call_args_list], [2, 2, 2]
        )
        zero_grad.assert_called_once_with(set_to_none=True)
        step.assert_called_once_with()
        self.assertEqual(len(losses), 3)
        for actual, expected in zip(
            losses, (policy_loss, value_loss, total_loss), strict=True
        ):
            self.assertAlmostEqual(actual, expected.item(), places=6)
        torch.testing.assert_close(self.model.state_dict(), reference.state_dict())
        torch.testing.assert_close(
            self.optimizer.state_dict(), reference_optimizer.state_dict()
        )

    def test_second_mini_batch_failure_clears_gradients_for_retry(self) -> None:
        before = deepcopy(self.model.state_dict())
        reference = deepcopy(self.model)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.03, momentum=0.5
        )
        first, second = (
            self.model(torch.from_numpy(self.samples.features[start : start + 2]))
            for start in (0, 2)
        )
        with patch.object(
            self.model,
            "forward",
            side_effect=[first, (second[0], second[1].squeeze(-1))],
        ):
            with self.assertRaisesRegex(ValueError, "value prediction shape"):
                self.accumulate()

        self.assertTrue(
            all(parameter.grad is None for parameter in self.model.parameters())
        )
        self.assertFalse(self.optimizer.state)
        torch.testing.assert_close(self.model.state_dict(), before, rtol=0, atol=0)

        self.accumulate()
        train_accumulated_batch(
            reference,
            reference_optimizer,
            self.samples,
            device=torch.device("cpu"),
            mini_batch_size=2,
            value_loss_weight=0.25,
        )
        torch.testing.assert_close(
            self.model.state_dict(), reference.state_dict(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
