import unittest

import torch
from torch.nn.functional import rms_norm

from ..positional_encoding.axial_rope_2d import AxialRoPE2D
from .encoder_block import EncoderBlock


class EncoderBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model_dim = 32
        self.num_heads = 4
        self.ffn_hidden_dim = 48
        self.block = self._make_block(board_size=9)
        self.tokens = torch.randn(2, 82, self.model_dim)

    def _make_block(self, board_size: int) -> EncoderBlock:
        rope = AxialRoPE2D(board_size, self.model_dim // self.num_heads)
        return EncoderBlock(self.model_dim, self.num_heads, self.ffn_hidden_dim, rope)

    @torch.no_grad()
    def test_forward_matches_pre_rmsnorm_reference(self) -> None:
        attention_weight = self.block.attention_norm.weight
        ffn_weight = self.block.ffn_norm.weight
        self.assertIsNot(attention_weight, ffn_weight)
        attention_weight.fill_(0.75)
        ffn_weight.fill_(1.25)

        normalized_shape = (self.model_dim,)
        attention_input = rms_norm(self.tokens, normalized_shape, attention_weight)
        after_attention = self.tokens + self.block.attention(attention_input)
        ffn_input = rms_norm(after_attention, normalized_shape, ffn_weight)
        expected = after_attention + self.block.ffn(ffn_input)

        output = self.block(self.tokens)
        self.assertEqual(output.shape, self.tokens.shape)
        torch.testing.assert_close(output, expected)

    def test_backward_on_cpu(self) -> None:
        self._assert_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_on_cuda(self) -> None:
        self._assert_backward("cuda")

    def _assert_backward(self, device: str) -> None:
        block = self.block.to(device)
        tokens = self.tokens.to(device).requires_grad_()
        output = block(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()

        for name, tensor in (("tokens", tokens), *block.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    @torch.no_grad()
    def test_reuses_weights_on_larger_board(self) -> None:
        larger_block = self._make_block(board_size=13)
        larger_block.load_state_dict(self.block.state_dict())
        tokens = torch.randn(2, 170, self.model_dim)

        output = larger_block(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_rejects_incompatible_token_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 82 tokens"):
            self.block(self.tokens[:, :-1])


if __name__ == "__main__":
    unittest.main()
