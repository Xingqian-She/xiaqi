import math
import unittest

import torch

from src.model.attention.multi_head_self_attention import MultiHeadSelfAttention
from src.model.positional_encoding.axial_rope_2d import AxialRoPE2D


class MultiHeadSelfAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.num_heads = 4
        self.head_dim = 8
        self.model_dim = self.num_heads * self.head_dim
        self.rope = AxialRoPE2D(board_size=9, head_dim=self.head_dim)
        self.attention = MultiHeadSelfAttention(self.model_dim, self.num_heads, self.rope)
        self.tokens = torch.randn(2, 82, self.model_dim)

    @torch.no_grad()
    def test_forward_matches_bidirectional_reference(self) -> None:
        output = self.attention(self.tokens)

        projections = [
            self.tokens @ weight.T for weight in self.attention.qkv_proj.weight.chunk(3, dim=0)
        ]
        query, key, value = (
            projection.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
            for projection in projections
        )
        query, key = self.rope(query, key)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attended = scores.softmax(dim=-1) @ value
        expected = attended.transpose(1, 2).flatten(-2) @ self.attention.out_proj.weight.T

        self.assertEqual(output.shape, self.tokens.shape)
        self.assertIsNone(self.attention.qkv_proj.bias)
        self.assertIsNone(self.attention.out_proj.bias)
        torch.testing.assert_close(output, expected)

    def test_backward_on_cpu(self) -> None:
        self._assert_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_on_cuda(self) -> None:
        self._assert_backward("cuda")

    def _assert_backward(self, device: str) -> None:
        attention = self.attention.to(device)
        tokens = self.tokens.to(device).requires_grad_()
        output = attention(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()

        for name, tensor in (("tokens", tokens), *attention.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    def test_rejects_incompatible_model_dimensions(self) -> None:
        for model_dim, num_heads, message in (
            (0, 4, "must be positive"),
            (32, 0, "must be positive"),
            (31, 4, "must be divisible"),
            (16, 4, r"expected rope\.head_dim=4"),
        ):
            with self.subTest(model_dim=model_dim, num_heads=num_heads):
                with self.assertRaisesRegex(ValueError, message):
                    MultiHeadSelfAttention(model_dim, num_heads, self.rope)

    def test_rejects_incompatible_input_shapes(self) -> None:
        for tokens, message in (
            (self.tokens[0], "tokens must have shape"),
            (self.tokens[..., :-1], "tokens must have shape"),
            (self.tokens[:, :-1], "expected 82 tokens"),
        ):
            with self.subTest(shape=tokens.shape):
                with self.assertRaisesRegex(ValueError, message):
                    self.attention(tokens)


if __name__ == "__main__":
    unittest.main()
