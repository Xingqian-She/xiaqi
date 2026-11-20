import math
import os
import unittest
from copy import deepcopy

import torch

from ..positional_encoding.axial_rope_2d import AxialRoPE2D
from .encoder import Encoder
from .encoder_block import EncoderBlock


class EncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model_dim = 32
        self.num_heads = 4
        self.ffn_hidden_dim = 48
        self.rope = AxialRoPE2D(board_size=9, head_dim=self.model_dim // self.num_heads)
        self.tokens = torch.randn(2, 82, self.model_dim)

    def _make_encoder(self, rope: AxialRoPE2D, num_layers: int = 3) -> Encoder:
        return Encoder(self.model_dim, self.num_heads, self.ffn_hidden_dim, num_layers, rope)

    @torch.no_grad()
    def test_forward_matches_stacked_blocks(self) -> None:
        encoder = self._make_encoder(self.rope)
        first, second, third = encoder.blocks

        output = encoder(self.tokens)
        self.assertEqual(output.shape, self.tokens.shape)
        torch.testing.assert_close(output, third(second(first(self.tokens))))

    def test_registers_independent_blocks_with_shared_rope(self) -> None:
        encoder = self._make_encoder(self.rope)
        self.assertEqual(len(encoder.blocks), 3)
        for block in encoder.blocks:
            self.assertIs(block.attention.rope, self.rope)

        parameter_count = sum(len(list(block.parameters())) for block in encoder.blocks)
        self.assertEqual(len(list(encoder.parameters())), parameter_count)

    def test_scales_only_residual_output_projections(self) -> None:
        projection_names = {"attention.out_proj.weight", "ffn.out_proj.weight"}
        for num_layers in (1, 3):
            torch.manual_seed(0)
            encoder = self._make_encoder(self.rope, num_layers)
            self.assertEqual(len(encoder.blocks), num_layers)
            torch.manual_seed(0)
            scale = 1.0 / math.sqrt(2 * num_layers)

            for layer, block in enumerate(encoder.blocks):
                reference = EncoderBlock(
                    self.model_dim, self.num_heads, self.ffn_hidden_dim, self.rope
                )
                expected = {
                    name: weight * scale if name in projection_names else weight
                    for name, weight in reference.state_dict().items()
                }
                with self.subTest(num_layers=num_layers, layer=layer):
                    torch.testing.assert_close(block.state_dict(), expected, rtol=0, atol=0)

    def test_backward_on_cpu(self) -> None:
        self._assert_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_on_cuda(self) -> None:
        self._assert_backward("cuda")

    def _assert_backward(self, device: str) -> None:
        encoder = self._make_encoder(self.rope).to(device)
        tokens = self.tokens.to(device).requires_grad_()
        output = encoder(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()

        for name, tensor in (("tokens", tokens), *encoder.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    @torch.no_grad()
    def test_loads_weights_on_larger_board(self) -> None:
        weights = self._make_encoder(self.rope).state_dict()
        larger_rope = AxialRoPE2D(board_size=13, head_dim=self.rope.head_dim)
        larger_encoder = self._make_encoder(larger_rope)
        larger_encoder.load_state_dict(weights)
        torch.testing.assert_close(larger_encoder.state_dict(), weights, rtol=0, atol=0)

        tokens = torch.randn(2, 170, self.model_dim)
        output = larger_encoder(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_rejects_nonpositive_layer_count(self) -> None:
        for num_layers in (0, -1):
            with self.subTest(num_layers=num_layers):
                with self.assertRaisesRegex(ValueError, "num_layers must be positive"):
                    self._make_encoder(self.rope, num_layers)

    @unittest.skipUnless(
        os.environ.get("XIAQI_TEST_ENCODER_COMPILE") == "1",
        "Set XIAQI_TEST_ENCODER_COMPILE=1 in the CUDA dev container",
    )
    def test_cuda_compile_matches_eager(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "CUDA is required for the Inductor check")
        eager_encoder = self._make_encoder(self.rope).to("cuda")
        compiled_encoder = deepcopy(eager_encoder)
        eager_tokens = self.tokens.to("cuda").requires_grad_()
        compiled_tokens = eager_tokens.detach().clone().requires_grad_()

        eager_output = eager_encoder(eager_tokens)
        eager_output.square().mean().backward()

        compiled = torch.compile(compiled_encoder, backend="inductor", fullgraph=True)
        compiled_output = compiled(compiled_tokens)
        compiled_output.square().mean().backward()

        tolerances = {"rtol": 1e-4, "atol": 1e-5}
        torch.testing.assert_close(compiled_output, eager_output, **tolerances)
        self.assertIsNotNone(eager_tokens.grad)
        self.assertIsNotNone(compiled_tokens.grad)
        torch.testing.assert_close(compiled_tokens.grad, eager_tokens.grad, **tolerances)
        for name, parameter in eager_encoder.named_parameters():
            with self.subTest(parameter=name):
                compiled_parameter = compiled_encoder.get_parameter(name)
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(compiled_parameter.grad)
                torch.testing.assert_close(compiled_parameter.grad, parameter.grad, **tolerances)


if __name__ == "__main__":
    unittest.main()
