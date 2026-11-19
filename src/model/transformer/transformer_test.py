import os
import unittest
from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn.functional import silu

from .transformer import Transformer


class TransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.in_channels = 3
        self.model = self._make_model()
        self.board = torch.randn(2, self.in_channels, 9, 9)

    def _make_model(
        self, board_size: int = 9, num_heads: int = 4, value_hidden_dim: int = 16
    ) -> Transformer:
        return Transformer(
            board_size=board_size,
            in_channels=self.in_channels,
            model_dim=32,
            num_heads=num_heads,
            ffn_hidden_dim=48,
            num_layers=3,
            value_hidden_dim=value_hidden_dim,
        )

    def _assert_outputs(
        self, outputs: tuple[Tensor, Tensor], batch_size: int, board_size: int
    ) -> None:
        policy_logits, value = outputs
        self.assertEqual(policy_logits.shape, (batch_size, board_size**2 + 1))
        self.assertEqual(value.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(policy_logits).all().item())
        self.assertTrue(value.abs().le(1).all().item())

    @torch.no_grad()
    def test_forward_matches_stem_encoder_and_heads(self) -> None:
        model = self.model
        features = model.stem(self.board)
        batch_size, model_dim, _, _ = features.shape
        board_tokens = features.permute(0, 2, 3, 1).reshape(batch_size, -1, model_dim)
        cls_token = model.cls_token.expand(batch_size, 1, model_dim)
        encoded = model.encoder(torch.cat((cls_token, board_tokens), dim=1))
        cls = encoded[:, 0]

        expected_points = (encoded[:, 1:] @ model.point_head.weight.T).squeeze(-1)
        expected_pass = cls @ model.pass_head.weight.T
        value_hidden = silu(cls @ model.value_head[0].weight.T)
        expected_value = (value_hidden @ model.value_head[2].weight.T).tanh()

        outputs = model(self.board)
        self._assert_outputs(outputs, batch_size=batch_size, board_size=9)
        policy_logits, value = outputs
        torch.testing.assert_close(policy_logits[:, :-1], expected_points)
        torch.testing.assert_close(policy_logits[:, -1:], expected_pass)
        torch.testing.assert_close(value, expected_value)

    def test_conv_and_linear_layers_are_bias_free(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                with self.subTest(module=name):
                    self.assertIsNone(module.bias)

    def test_backward_on_cpu(self) -> None:
        self._assert_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_on_cuda(self) -> None:
        self._assert_backward("cuda")

    def _assert_backward(self, device: str) -> None:
        model = self.model.to(device)
        board = self.board.to(device).requires_grad_()
        outputs = model(board)
        self._assert_outputs(outputs, batch_size=board.size(0), board_size=9)
        policy_logits, value = outputs
        (policy_logits.square().mean() + value.square().mean()).backward()

        for name, tensor in (("board", board), *model.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    @torch.no_grad()
    def test_loads_weights_on_larger_board(self) -> None:
        weights = self.model.state_dict()
        larger_model = self._make_model(board_size=13)
        larger_model.load_state_dict(weights)
        torch.testing.assert_close(larger_model.state_dict(), weights, rtol=0, atol=0)

        board = torch.randn(1, self.in_channels, 13, 13)
        self._assert_outputs(larger_model(board), batch_size=1, board_size=13)

    def test_rejects_boards_with_wrong_spatial_size(self) -> None:
        for height, width in ((13, 13), (3, 27)):
            with self.subTest(height=height, width=width):
                board = torch.empty(2, self.in_channels, height, width)
                with self.assertRaisesRegex(ValueError, "board must have shape"):
                    self.model(board)

    def test_rejects_invalid_head_dimensions(self) -> None:
        for num_heads, value_hidden_dim, message in (
            (0, 16, "num_heads"),
            (3, 16, "num_heads"),
            (4, 0, "value_hidden_dim"),
        ):
            with self.subTest(num_heads=num_heads, value_hidden_dim=value_hidden_dim):
                with self.assertRaisesRegex(ValueError, message):
                    self._make_model(num_heads=num_heads, value_hidden_dim=value_hidden_dim)

    @unittest.skipUnless(
        os.environ.get("XIAQI_TEST_TRANSFORMER_COMPILE") == "1",
        "Set XIAQI_TEST_TRANSFORMER_COMPILE=1 in the CUDA dev container",
    )
    def test_cuda_compile_matches_eager(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "CUDA is required for the Inductor check")
        eager_model = self.model.to("cuda")
        compiled_model = deepcopy(eager_model)
        eager_board = self.board.to("cuda").requires_grad_()
        compiled_board = eager_board.detach().clone().requires_grad_()

        eager_logits, eager_value = eager_model(eager_board)
        (eager_logits.square().mean() + eager_value.square().mean()).backward()

        compiled = torch.compile(compiled_model, backend="inductor", fullgraph=True)
        compiled_logits, compiled_value = compiled(compiled_board)
        (compiled_logits.square().mean() + compiled_value.square().mean()).backward()

        tolerances = {"rtol": 1e-4, "atol": 1e-5}
        torch.testing.assert_close(compiled_logits, eager_logits, **tolerances)
        torch.testing.assert_close(compiled_value, eager_value, **tolerances)
        self.assertIsNotNone(eager_board.grad)
        self.assertIsNotNone(compiled_board.grad)
        torch.testing.assert_close(compiled_board.grad, eager_board.grad, **tolerances)
        for name, parameter in eager_model.named_parameters():
            with self.subTest(parameter=name):
                compiled_parameter = compiled_model.get_parameter(name)
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(compiled_parameter.grad)
                torch.testing.assert_close(compiled_parameter.grad, parameter.grad, **tolerances)


if __name__ == "__main__":
    unittest.main()
