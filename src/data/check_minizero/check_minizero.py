"""Smoke-test the MiniZero Python binding with the project's 9x9 Go config."""

from pathlib import Path
from types import ModuleType

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[3] / "runs/go-9x9-az/minizero_base.cfg"
BOARD_SIZE = 9
BLACK, WHITE = 1, 2
PASS = BOARD_SIZE**2
INPUT_SHAPE = (18, BOARD_SIZE, BOARD_SIZE)
ACTION_FEATURE_SHAPE = (1, BOARD_SIZE, BOARD_SIZE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_array(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    _require(
        actual.dtype == np.float32 and actual.flags.c_contiguous,
        f"{name} should return a C-contiguous float32 array.",
    )
    np.testing.assert_array_equal(actual, expected, err_msg=name)


def check_go_environment(binding: ModuleType) -> None:
    """Run a fixed capture and double pass, then check that reset restores the board."""
    _require(
        hasattr(binding, "Environment"), "Rebuild minizero_py with Environment support."
    )
    _require(binding.load_config_file(str(CONFIG_PATH)), f"Cannot load {CONFIG_PATH}.")
    env = binding.Environment()

    game = env.name()
    _require(game == "go_9x9", f"Expected go_9x9, got {game}.")
    _require(env.num_players() == 2, "Go should have two players.")
    _require(env.policy_size() == PASS + 1, "Expected 81 board points and one pass.")
    _require(tuple(env.input_shape()) == INPUT_SHAPE, "Unexpected input_shape().")
    _require(
        tuple(env.action_feature_shape()) == ACTION_FEATURE_SHAPE,
        "Unexpected action_feature_shape().",
    )

    _require(env.turn() == BLACK, "Black should play first.")
    _require(
        not env.is_terminal() and env.action_history() == [], "Expected a new game."
    )
    initial_actions = sorted(env.legal_actions())
    _require(
        initial_actions == list(range(PASS + 1)),
        "Expected all points and pass to be legal.",
    )

    initial_features = env.features()
    expected_initial = np.zeros(INPUT_SHAPE, dtype=np.float32)
    expected_initial[16] = 1.0  # Black to play.
    _check_array("Empty-board features", initial_features, expected_initial)

    b1, a1, a2 = 1, 0, BOARD_SIZE
    action_features = env.action_features(b1)
    expected_action = np.zeros(ACTION_FEATURE_SHAPE, dtype=np.float32)
    expected_action[0, 0, 1] = 1.0
    _check_array("B1 action features", action_features, expected_action)
    _check_array(
        "Pass action features",
        env.action_features(PASS),
        np.zeros_like(expected_action),
    )
    _check_array(
        "Rotated B1 action features",
        env.action_features(b1, rotation=1),
        np.rot90(expected_action, axes=(1, 2)),
    )
    print(
        f"Game={game}, players={env.num_players()}, policy_size={env.policy_size()}, pass={PASS}"
    )
    print(
        f"features={initial_features.shape} {initial_features.dtype}, "
        f"action_features={action_features.shape} {action_features.dtype}"
    )

    _require(
        env.is_legal_action(b1) and env.act(b1), "Black should be able to play B1."
    )
    _require(
        b1 not in env.legal_actions()
        and not env.is_legal_action(b1)
        and not env.act(b1),
        "An occupied point should be illegal and act() should return False.",
    )
    _require(
        env.turn() == WHITE and env.action_history() == [(b1, BLACK)],
        "The rejected move should preserve the turn and history.",
    )
    expected_after_b1 = np.zeros_like(expected_initial)
    expected_after_b1[1, 0, 1] = 1.0  # Black is the opponent from White's perspective.
    expected_after_b1[17] = 1.0
    _check_array("Board features after B1", env.features(), expected_after_b1)
    _check_array(
        "Rotated board features after B1",
        env.features(rotation=1),
        np.rot90(expected_after_b1, axes=(1, 2)),
    )

    _require(env.act(a1), "White should be able to play A1.")
    _require(env.act(a2), "Black should be able to capture A1 by playing A2.")
    after_capture = env.features()
    _require(
        after_capture[0, 0, 0] == 0.0 and after_capture[2, 0, 0] == 1.0,
        "The white stone at A1 should appear in the previous position only.",
    )
    _require(
        env.act(PASS) and not env.is_terminal(), "One pass should keep the game going."
    )
    _require(
        env.act(PASS) and env.is_terminal(),
        "Two consecutive passes should end the game.",
    )
    history = env.action_history()
    expected_history = [
        (b1, BLACK),
        (a1, WHITE),
        (a2, BLACK),
        (PASS, WHITE),
        (PASS, BLACK),
    ]
    _require(history == expected_history, f"Unexpected action history {history}.")
    score, reward = env.eval_score(), env.reward()
    _require(
        env.turn() == WHITE and score == 1.0, "Expected a Black win with White to play."
    )
    _require(reward == 0.0, "Go reward() should be zero at the terminal position.")
    print(f"History={history}")
    print(
        f"Terminal turn={env.turn()}, reward={reward:g}, eval_score={score:+.0f} (Black wins)"
    )
    print(f"Pass legal after terminal={env.is_legal_action(PASS)}")

    _check_array("Saved empty-board features", initial_features, expected_initial)
    env.reset()
    _require(
        env.turn() == BLACK and not env.is_terminal(),
        "Reset should restart at Black's turn.",
    )
    _require(env.action_history() == [], "Reset should clear move history.")
    _require(
        sorted(env.legal_actions()) == initial_actions,
        "Reset should restore all legal actions.",
    )
    _check_array("Board features after reset", env.features(), initial_features)
    print("MiniZero Go environment check passed.")


def main() -> None:
    try:
        import minizero_py
    except ImportError as error:
        raise SystemExit(
            f"Cannot import minizero_py ({error}). "
            "Build with scripts/build_minizero.sh --python and set PYTHONPATH "
            "as shown in src/data/check_minizero/README.md."
        ) from error

    print(f"Binding={minizero_py.__file__}")
    print(f"Config={CONFIG_PATH}")
    check_go_environment(minizero_py)


if __name__ == "__main__":
    main()
