import unittest

import torch
from torch import nn

from .ema import ExponentialMovingAverage


class ExponentialMovingAverageTest(unittest.TestCase):
    @torch.no_grad()
    def test_averages_parameters_and_copies_buffers_after_each_update(self) -> None:
        source = nn.Linear(2, 1, bias=False)
        source.weight.zero_()
        source.register_buffer("running_value", torch.tensor(0.0))
        source.register_buffer("counter", torch.tensor(0), persistent=False)
        average = ExponentialMovingAverage(source, tau=0.25)

        self.assertFalse(average.model.training)
        self.assertFalse(average.model.weight.requires_grad)
        self.assertTrue(source.training)
        self.assertTrue(source.weight.requires_grad)
        self.assertNotEqual(source.weight.data_ptr(), average.model.weight.data_ptr())

        for step, (current, expected) in enumerate(((4.0, 1.0), (9.0, 3.0)), start=1):
            source.weight.fill_(current)
            source.running_value.fill_(step * 2.0)
            source.counter.fill_(step)
            average.update()
            torch.testing.assert_close(
                average.model.weight, torch.full_like(source.weight, expected)
            )
            torch.testing.assert_close(
                average.model.running_value, source.running_value
            )
            torch.testing.assert_close(average.model.counter, source.counter)

    def test_tau_one_copies_current_weights(self) -> None:
        source = nn.Linear(2, 1)
        average = ExponentialMovingAverage(source, tau=1.0)
        with torch.no_grad():
            source.weight.fill_(1.0)
            source.bias.fill_(2.0)
        average.update()
        torch.testing.assert_close(
            average.model.state_dict(), source.state_dict(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
