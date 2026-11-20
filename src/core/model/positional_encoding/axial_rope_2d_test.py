import math
import unittest

import torch

from .axial_rope_2d import AxialRoPE2D


class AxialRoPE2DTest(unittest.TestCase):
    def test_cls_half_turn_and_center_zero_phase(self) -> None:
        board_size = 9
        tokens = 1 + board_size**2
        center_token = 1 + board_size**2 // 2
        rope = AxialRoPE2D(board_size=board_size, head_dim=8)
        query = torch.arange(1.0, 9.0).expand(1, 1, tokens, 8)
        key = torch.arange(8.0, 0.0, -1.0).expand_as(query)

        rotated_query, rotated_key = rope(query, key)

        for values, rotated in ((query, rotated_query), (key, rotated_key)):
            torch.testing.assert_close(rotated[..., 0, :], -values[..., 0, :])
            torch.testing.assert_close(rotated[..., center_token, :], values[..., center_token, :])

    def test_signed_axis_rotations_and_frequencies(self) -> None:
        board_size = 9
        tokens = 1 + board_size**2
        center_token = 1 + board_size**2 // 2
        values = torch.tensor([1.0, 0.0] * 4).expand(1, 1, tokens, 8)

        for rope, second_frequency in (
            (AxialRoPE2D(board_size=board_size, head_dim=8), 0.1),
            (AxialRoPE2D(board_size=board_size, head_dim=8, theta=16.0), 0.25),
        ):
            rotated, _ = rope(values, values)
            for row, col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                with self.subTest(row=row, col=col, second_frequency=second_frequency):
                    token = center_token + row * board_size + col
                    angles = (col, col * second_frequency, row, row * second_frequency)
                    expected = values.new_tensor(
                        [part for angle in angles for part in (math.cos(angle), math.sin(angle))]
                    )
                    torch.testing.assert_close(rotated[0, 0, token], expected)

    def test_board_dot_products_depend_on_relative_offset(self) -> None:
        delta_row, delta_col = 1, -2
        expected = torch.tensor(math.sin(delta_row) - math.sin(delta_col))
        for board_size in (9, 13, 19):
            rope = AxialRoPE2D(board_size=board_size, head_dim=4)
            tokens = 1 + board_size**2
            query = torch.tensor([1.0, 0.0, 0.0, 1.0]).expand(1, 1, tokens, 4)
            key = torch.tensor([0.0, 1.0, 1.0, 0.0]).expand_as(query)
            rotated_query, rotated_key = rope(query, key)

            center = board_size // 2
            for row, col in ((1, 2), (center, center)):
                with self.subTest(board_size=board_size, row=row, col=col):
                    query_token = 1 + row * board_size + col
                    key_token = query_token + delta_row * board_size + delta_col
                    score = rotated_query[0, 0, query_token] @ rotated_key[0, 0, key_token]
                    torch.testing.assert_close(score, expected)

    def test_preserves_shape_and_norm_and_backpropagates(self) -> None:
        rope = AxialRoPE2D(board_size=9, head_dim=8)
        generator = torch.Generator().manual_seed(0)
        query = torch.randn(2, 82, 3, 8, generator=generator).transpose(1, 2).requires_grad_()
        key = torch.randn(2, 82, 3, 8, generator=generator).transpose(1, 2).requires_grad_()
        rotated_query, rotated_key = rope(query, key)

        for values, rotated in ((query, rotated_query), (key, rotated_key)):
            self.assertEqual(rotated.shape, values.shape)
            torch.testing.assert_close(rotated.norm(dim=-1), values.norm(dim=-1))

        (rotated_query.square().sum() + rotated_key.square().sum()).backward()
        torch.testing.assert_close(query.grad, 2 * query)
        torch.testing.assert_close(key.grad, 2 * key)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_preserves_dtype_and_matches_cpu(self) -> None:
        rope = AxialRoPE2D(board_size=9, head_dim=8)
        query = torch.linspace(-1, 1, 82 * 8, dtype=torch.float16).reshape(1, 1, 82, 8)
        key = query.flip(-1)
        expected = rope(query, key)
        actual = rope.to("cuda")(query.to("cuda"), key.to("cuda"))

        for rotated, reference in zip(actual, expected):
            self.assertEqual(rotated.device.type, "cuda")
            self.assertEqual(rotated.dtype, torch.float16)
            torch.testing.assert_close(rotated.cpu(), reference)

    def test_rotation_tables_are_not_saved_in_state_dict(self) -> None:
        rope = AxialRoPE2D(board_size=9, head_dim=8)
        self.assertEqual(rope.state_dict(), {})

    def test_rejects_invalid_configuration(self) -> None:
        for parameter, board_size, head_dim, theta in (
            ("board_size", -1, 8, 100.0),
            ("board_size", 8, 8, 100.0),
            ("head_dim", 9, 0, 100.0),
            ("head_dim", 9, 6, 100.0),
            ("theta", 9, 8, 0.0),
            ("theta", 9, 8, math.inf),
            ("theta", 9, 8, math.nan),
        ):
            with self.subTest(board_size=board_size, head_dim=head_dim, theta=theta):
                with self.assertRaisesRegex(ValueError, parameter):
                    AxialRoPE2D(board_size=board_size, head_dim=head_dim, theta=theta)

    def test_rejects_invalid_inputs(self) -> None:
        rope = AxialRoPE2D(board_size=9, head_dim=8)
        values = torch.zeros(2, 3, 82, 8)
        wrong_tokens = values[:, :, :-1]
        wrong_width = values[..., :-1]
        integer_values = values.long()

        for case, query, key, message in (
            ("rank", values[0], values[0], "matching"),
            ("different shapes", values, values[:, :1], "matching"),
            ("token count", wrong_tokens, wrong_tokens, "expected 82 tokens"),
            ("head width", wrong_width, wrong_width, "head_dim=8"),
            ("integer query", integer_values, values, "floating-point"),
            ("integer key", values, integer_values, "floating-point"),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, message):
                    rope(query, key)


if __name__ == "__main__":
    unittest.main()
