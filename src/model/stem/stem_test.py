import unittest

import torch

from .stem import Stem


class StemTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.in_channels = 3
        self.model_dim = 8
        self.stem = Stem(self.in_channels, self.model_dim)

    @torch.no_grad()
    def test_projects_channels_and_preserves_board_size(self) -> None:
        self.assertEqual(
            self.stem.conv.weight.shape, (self.model_dim, self.in_channels, 3, 3)
        )
        self.assertIsNone(self.stem.conv.bias)

        for batch_size, board_size in ((1, 9), (2, 13)):
            with self.subTest(batch_size=batch_size, board_size=board_size):
                board = torch.randn(batch_size, self.in_channels, board_size, board_size)
                output = self.stem(board)
                self.assertEqual(
                    output.shape, (batch_size, self.model_dim, board_size, board_size)
                )

    @torch.no_grad()
    def test_zero_padding_at_board_edges(self) -> None:
        stem = Stem(in_channels=1, model_dim=1)
        stem.conv.weight.fill_(1.0)
        board = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
        expected = torch.tensor(
            [[12.0, 21.0, 16.0], [27.0, 45.0, 33.0], [24.0, 39.0, 28.0]]
        )

        torch.testing.assert_close(stem(board)[0, 0], expected)

    def test_backward_on_cpu(self) -> None:
        self._assert_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_on_cuda(self) -> None:
        self._assert_backward("cuda")

    def _assert_backward(self, device: str) -> None:
        stem = self.stem.to(device)
        board = torch.randn(2, self.in_channels, 9, 9, device=device, requires_grad=True)
        output = stem(board)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()

        for name, tensor in (("board", board), *stem.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    def test_rejects_nonpositive_dimensions(self) -> None:
        for in_channels, model_dim in (
            (0, self.model_dim),
            (-1, self.model_dim),
            (self.in_channels, 0),
            (self.in_channels, -1),
        ):
            with self.subTest(in_channels=in_channels, model_dim=model_dim):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    Stem(in_channels, model_dim)

    def test_rejects_incompatible_input_shapes(self) -> None:
        for shape in (
            (self.in_channels, 9, 9),
            (2, self.in_channels + 1, 9, 9),
            (2, self.in_channels, 9, 13),
            (2, self.in_channels, 0, 0),
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "board must have shape"):
                    self.stem(torch.empty(shape))


if __name__ == "__main__":
    unittest.main()
