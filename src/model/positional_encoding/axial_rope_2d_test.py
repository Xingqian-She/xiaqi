import math
import unittest

import torch

from axial_rope_2d import AxialRoPE2D


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

    def test_one_step_right_and_down_rotate_different_pairs(self) -> None:
        board_size = 9
        tokens = 1 + board_size**2
        center_token = 1 + board_size**2 // 2
        rope = AxialRoPE2D(board_size=board_size, head_dim=4)
        values = torch.tensor([1.0, 0.0, 1.0, 0.0]).expand(1, 1, tokens, 4)

        rotated, _ = rope(values, values)

        cos, sin = math.cos(1), math.sin(1)
        right = values.new_tensor([cos, sin, 1.0, 0.0])
        down = values.new_tensor([1.0, 0.0, cos, sin])
        torch.testing.assert_close(rotated[0, 0, center_token + 1], right)
        torch.testing.assert_close(rotated[0, 0, center_token + board_size], down)


if __name__ == "__main__":
    unittest.main()
