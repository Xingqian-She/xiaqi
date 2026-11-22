import unittest

import torch

from .residual_conv_stack import ResidualConvStack


class ResidualConvStackTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model_dim = 8
        self.features = torch.randn(2, self.model_dim, 9, 9)

    @torch.no_grad()
    def test_applies_independent_blocks_in_order(self) -> None:
        stack = ResidualConvStack(self.model_dim, num_blocks=2)
        first, second = stack.blocks
        self.assertIsNot(first.conv.weight, second.conv.weight)
        self.assertIsNot(first.norm.weight, second.norm.weight)

        after_first = self.features + first(self.features)
        expected = after_first + second(after_first)
        torch.testing.assert_close(stack(self.features), expected)

    def test_zero_blocks_returns_input(self) -> None:
        stack = ResidualConvStack(self.model_dim, num_blocks=0)

        self.assertIs(stack(self.features), self.features)
        self.assertEqual(stack.state_dict(), {})

    def test_rejects_invalid_block_counts(self) -> None:
        for num_blocks in (-1, 1.5, True):
            with self.subTest(num_blocks=num_blocks):
                with self.assertRaisesRegex(
                    ValueError, "num_blocks must be a nonnegative integer"
                ):
                    ResidualConvStack(self.model_dim, num_blocks)


if __name__ == "__main__":
    unittest.main()
