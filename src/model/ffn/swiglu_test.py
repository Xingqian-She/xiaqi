import unittest

import torch

from .swiglu import SwiGLU


class SwiGLUTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model_dim = 32
        self.hidden_dim = 48
        self.ffn = SwiGLU(self.model_dim, self.hidden_dim)
        self.tokens = torch.randn(2, 82, self.model_dim)

    @torch.no_grad()
    def test_forward_matches_swiglu_reference(self) -> None:
        self.assertEqual(
            self.ffn.gate_up_proj.weight.shape, (2 * self.hidden_dim, self.model_dim)
        )
        self.assertEqual(
            self.ffn.out_proj.weight.shape, (self.model_dim, self.hidden_dim)
        )
        self.assertIsNone(self.ffn.gate_up_proj.bias)
        self.assertIsNone(self.ffn.out_proj.bias)

        gate_weight, up_weight = self.ffn.gate_up_proj.weight.split(
            self.hidden_dim, dim=0
        )
        for tokens in (
            self.tokens,
            torch.randn(2, 170, self.model_dim),
            self.tokens[0, 0],
        ):
            with self.subTest(shape=tokens.shape):
                gate = tokens @ gate_weight.T
                up = tokens @ up_weight.T
                expected = (gate * gate.sigmoid() * up) @ self.ffn.out_proj.weight.T

                output = self.ffn(tokens)
                self.assertEqual(output.shape, tokens.shape)
                torch.testing.assert_close(output, expected)

    def test_backward_on_cpu(self) -> None:
        self._assert_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_backward_on_cuda(self) -> None:
        self._assert_backward("cuda")

    def _assert_backward(self, device: str) -> None:
        ffn = self.ffn.to(device)
        tokens = self.tokens.to(device).requires_grad_()
        output = ffn(tokens)
        self.assertTrue(torch.isfinite(output).all().item())
        output.square().mean().backward()

        for name, tensor in (("tokens", tokens), *ffn.named_parameters()):
            with self.subTest(tensor=name):
                gradient = tensor.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue(gradient.ne(0).any().item())

    def test_rejects_nonpositive_dimensions(self) -> None:
        for model_dim, hidden_dim in (
            (0, self.hidden_dim),
            (-1, self.hidden_dim),
            (self.model_dim, 0),
            (self.model_dim, -1),
        ):
            with self.subTest(model_dim=model_dim, hidden_dim=hidden_dim):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    SwiGLU(model_dim, hidden_dim)

    def test_rejects_incompatible_input_shapes(self) -> None:
        for tokens in (torch.tensor(0.0), self.tokens[..., :-1]):
            with self.subTest(shape=tokens.shape):
                with self.assertRaisesRegex(ValueError, "tokens must have shape"):
                    self.ffn(tokens)


if __name__ == "__main__":
    unittest.main()
