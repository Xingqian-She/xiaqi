import unittest

import torch
from torch.nn.functional import conv2d, silu

from .residual_conv_block import ResidualConvBlock


class ResidualConvBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model_dim = 8
        self.block = ResidualConvBlock(self.model_dim)

    @torch.no_grad()
    def test_forward_matches_channel_layer_norm_reference(self) -> None:
        self.assertIsNone(self.block.conv.bias)
        self.assertEqual(
            self.block.conv.weight.shape, (self.model_dim, self.model_dim, 3, 3)
        )
        self.block.norm.weight.copy_(torch.linspace(0.5, 1.5, self.model_dim))
        self.block.norm.bias.copy_(torch.linspace(-0.5, 0.5, self.model_dim))

        for batch_size, board_size in ((1, 9), (2, 13)):
            with self.subTest(batch_size=batch_size, board_size=board_size):
                features = torch.randn(
                    batch_size, self.model_dim, board_size, board_size
                )
                convolved = conv2d(features, self.block.conv.weight, padding=1)
                mean = convolved.mean(dim=1, keepdim=True)
                variance = convolved.var(dim=1, unbiased=False, keepdim=True)
                normalized = (convolved - mean) * (
                    variance + self.block.norm.eps
                ).rsqrt()
                weight = self.block.norm.weight.view(1, -1, 1, 1)
                bias = self.block.norm.bias.view(1, -1, 1, 1)

                output = self.block(features)
                self.assertEqual(output.shape, features.shape)
                torch.testing.assert_close(output, silu(normalized * weight + bias))

    def test_backward_on_cpu(self) -> None:
        features = torch.randn(2, self.model_dim, 9, 9, requires_grad=True)
        self.block(features).square().mean().backward()

        for name, tensor in (("features", features), *self.block.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    def test_rejects_incompatible_input_shapes(self) -> None:
        for shape in ((self.model_dim, 9, 9), (2, 1, 9, 9), (2, self.model_dim, 9, 0)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "features must have shape"):
                    self.block(torch.empty(shape))


if __name__ == "__main__":
    unittest.main()
