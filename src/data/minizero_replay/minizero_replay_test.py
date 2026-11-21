import unittest
from dataclasses import replace
from importlib import import_module

import numpy as np

from ..minizero_record.minizero_record import parse_record
from .minizero_replay import ReplayError, replay_record

GAME = (
    "(;GM[go_9x9]SZ[9]KM[7]RE[-1]DLEN[0-3]"
    ";B[1]P[1:3,81:1];W[80]P[80:2,81:2]"
    ";B[81]P[4:1,81:2];W[81]P[81:1])"
)
RESIGNATION = "(;GM[go_9x9]SZ[9]KM[7]RE[1];B[1]P[1:1])"


class ReplayInputTest(unittest.TestCase):
    def test_rejects_intermediate_record(self) -> None:
        record = parse_record(f"SelfPlay false 1 1 1 {RESIGNATION} #")
        with self.assertRaisesRegex(ReplayError, "finished game"):
            replay_record(record)

    def test_rejects_sample_range_outside_the_game(self) -> None:
        record = replace(parse_record(GAME), sample_range=range(5))
        with self.assertRaisesRegex(ReplayError, "sample_range"):
            replay_record(record)


class MiniZeroReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import_module("minizero_py")
        except ModuleNotFoundError as error:
            if error.name != "minizero_py":
                raise
            raise unittest.SkipTest(
                "Build minizero_py in dev and set PYTHONPATH."
            ) from error

    def test_builds_pre_move_features_and_alpha_zero_targets(self) -> None:
        samples = replay_record(parse_record(GAME))
        self.assertEqual(samples.features.shape, (4, 18, 9, 9))
        self.assertEqual(samples.policy.shape, (4, 82))
        self.assertEqual(samples.value.shape, (4, 1))
        for array in (samples.features, samples.policy, samples.value):
            self.assertEqual(array.dtype, np.float32)

        self.assertFalse(samples.features[0, :16].any())
        np.testing.assert_array_equal(samples.features[0, 16], 1.0)
        self.assertFalse(samples.features[1, 0].any())
        np.testing.assert_array_equal(np.argwhere(samples.features[1, 1]), [[0, 1]])
        np.testing.assert_array_equal(samples.features[1, 17], 1.0)

        expected_policy = np.zeros((4, 82), dtype=np.float32)
        expected_policy[0, [1, 81]] = [0.75, 0.25]
        expected_policy[1, [80, 81]] = [0.5, 0.5]
        expected_policy[2, [4, 81]] = [1 / 3, 2 / 3]
        expected_policy[3, 81] = 1.0
        np.testing.assert_allclose(samples.policy, expected_policy)
        np.testing.assert_array_equal(samples.value[:, 0], [-1, 1, -1, 1])

    def test_replays_unselected_moves_without_requiring_their_policy(self) -> None:
        partial_range = (
            GAME.replace("DLEN[0-3]", "DLEN[1-2]")
            .replace("B[1]P[1:3,81:1]", "B[1]")
            .replace("W[81]P[81:1]", "W[81]P[]")
        )
        full = replay_record(parse_record(GAME))
        selected = replay_record(parse_record(partial_range))
        np.testing.assert_array_equal(selected.features, full.features[1:3])
        np.testing.assert_array_equal(selected.policy, full.policy[1:3])
        np.testing.assert_array_equal(selected.value, full.value[1:3])

    def test_rejects_missing_or_empty_policy_at_a_selected_move(self) -> None:
        for policy in ("", "P[]"):
            with self.subTest(policy=policy):
                text = GAME.replace("W[80]P[80:2,81:2]", f"W[80]{policy}")
                with self.assertRaisesRegex(ReplayError, "Move 2 is missing P"):
                    replay_record(parse_record(text))

    def test_rejects_policy_mass_on_an_occupied_point(self) -> None:
        record = parse_record(GAME.replace("P[80:2,81:2]", "P[80:2,1:2]"))
        with self.assertRaisesRegex(ReplayError, "Move 2 has illegal P action 1"):
            replay_record(record)

    def test_rejects_invalid_weights_on_programmatically_constructed_records(
        self,
    ) -> None:
        record = parse_record(GAME)
        first, *later = record.moves
        cases = (
            (((1, 0.0),), "positive finite P weight sum"),
            (((1, float("inf")),), "positive finite P weight sum"),
            (((1, 1.0), (1, 2.0)), "repeated or negative P weight"),
            (((1, 2.0), (81, -1.0)), "repeated or negative P weight"),
        )
        for weights, message in cases:
            with self.subTest(weights=weights):
                moves = (replace(first, policy_weights=weights), *later)
                with self.assertRaisesRegex(ReplayError, f"Move 1 .*{message}"):
                    replay_record(replace(record, moves=moves))

    def test_rejects_invalid_moves_even_outside_sample_range(self) -> None:
        text = GAME.replace("DLEN[0-3]", "DLEN[2-3]")
        cases = (
            (text.replace(";W[80]", ";B[80]"), "Move 2 has player 1, expected 2"),
            (text.replace(";W[80]", ";W[1]"), "Move 2 has illegal action 1"),
            (text[:-1] + ";B[81])", "Move 5 follows a terminal position"),
        )
        for game, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ReplayError, message):
                    replay_record(parse_record(game))

    def test_rejects_a_result_that_disagrees_with_terminal_scoring(self) -> None:
        record = parse_record(GAME.replace("RE[-1]", "RE[1]"))
        with self.assertRaisesRegex(ReplayError, "disagrees with RE"):
            replay_record(record)

    def test_resignation_requires_an_explicit_completion_marker(self) -> None:
        with self.assertRaisesRegex(ReplayError, "explicit completion marker"):
            replay_record(parse_record(RESIGNATION))

        record = parse_record(f"SelfPlay true 1 1 1 {RESIGNATION} #")
        samples = replay_record(record)
        np.testing.assert_array_equal(samples.value, [[1]])

    def test_uses_recorded_board_size_and_komi(self) -> None:
        # One stone of each color and zero komi on a 13x13 board is a draw.
        record = parse_record(
            "(;GM[go_13x13]SZ[13]KM[0]RE[0]"
            ";B[0]P[0:1];W[168]P[168:1];B[169]P[169:1];W[169]P[169:1])"
        )
        samples = replay_record(record, ko_rule="situational")
        self.assertEqual(samples.features.shape, (4, 18, 13, 13))
        self.assertEqual(samples.policy.shape, (4, 170))
        np.testing.assert_array_equal(samples.policy[2:, 169], [1, 1])
        np.testing.assert_array_equal(samples.value, np.zeros((4, 1)))

    def test_allows_immediate_resignation_without_samples(self) -> None:
        record = parse_record(
            "SelfPlay true 0 0 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1]DLEN[0--1]) #"
        )
        samples = replay_record(record)
        self.assertEqual(samples.features.shape, (0, 18, 9, 9))
        self.assertEqual(samples.policy.shape, (0, 82))
        self.assertEqual(samples.value.shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
