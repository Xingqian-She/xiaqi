"""Replay a MiniZero Go record and build one game's AlphaZero training batch."""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..minizero_record.game_record.game_record import GameRecord
from ..minizero_record.move.move import Move
from .training_samples.training_samples import TrainingSamples


class ReplayError(ValueError):
    """A record could not be replayed into valid training samples."""


def _write_policy(
    target: NDArray[np.float32],
    move: Move,
    legal_actions: set[int],
    move_number: int,
) -> None:
    if not move.policy_weights:
        raise ReplayError(
            f"Move {move_number} is missing P or has empty P[]; "
            "a training sample requires policy weights."
        )

    total = sum(weight for _, weight in move.policy_weights)
    if not 0 < total < math.inf:
        raise ReplayError(
            f"Move {move_number} requires a positive finite P weight sum."
        )
    seen: set[int] = set()
    for action_id, weight in move.policy_weights:
        if action_id not in legal_actions:
            raise ReplayError(f"Move {move_number} has illegal P action {action_id}.")
        if action_id in seen or weight < 0:
            raise ReplayError(
                f"Move {move_number} has a repeated or negative P weight for {action_id}."
            )
        seen.add(action_id)
        target[action_id] = weight / total


def replay_record(
    record: GameRecord,
    *,
    ko_rule: Literal["positional", "situational"] = "positional",
) -> TrainingSamples:
    """Build pre-move samples at the record's DLEN indices.

    Each selected move must have policy weights in P.
    The ko rule comes from the source configuration. Call this function
    serially because MiniZero's configuration is process-global.
    """
    if record.is_complete is False:
        raise ReplayError("Training samples require a finished game.")
    if not 1 <= record.board_size <= 19 or not math.isfinite(record.komi):
        raise ReplayError("Replay requires a Go board size of 1 to 19 and finite komi.")
    if record.black_result not in (-1, 0, 1):
        raise ReplayError("Expected a black result of -1, 0 or +1.")
    if ko_rule not in ("positional", "situational"):
        raise ValueError("Go ko_rule must be positional or situational.")

    selected = record.sample_range
    if selected.step != 1 or not 0 <= selected.start <= selected.stop <= len(
        record.moves
    ):
        raise ReplayError(
            "sample_range must select consecutive indices within the moves."
        )

    try:
        import minizero_py
    except ImportError as error:
        raise ImportError(
            "Replay requires the Go minizero_py extension built in the dev container."
        ) from error

    config = (
        f"env_board_size={record.board_size}:env_go_komi={record.komi}:"
        f"env_go_ko_rule={ko_rule}"
    )
    if not minizero_py.load_config_string(config):
        raise RuntimeError(
            f"Cannot configure the MiniZero Go environment with {config}."
        )
    env = minizero_py.Environment()
    policy_size = env.policy_size()
    game_name = env.name()
    if (
        game_name != f"go_{record.board_size}x{record.board_size}"
        or policy_size != record.board_size**2 + 1
    ):
        raise RuntimeError(
            f"Unexpected MiniZero environment {game_name} with {policy_size} actions."
        )

    sample_count = len(selected)
    features = np.empty((sample_count, *env.input_shape()), dtype=np.float32)
    policy = np.zeros((sample_count, policy_size), dtype=np.float32)
    value = np.empty((sample_count, 1), dtype=np.float32)

    for index, move in enumerate(record.moves):
        move_number = index + 1
        if env.is_terminal():
            raise ReplayError(f"Move {move_number} follows a terminal position.")
        player = env.turn()
        if move.player != player:
            raise ReplayError(
                f"Move {move_number} has player {move.player}, expected {player}."
            )
        if not 0 <= move.action_id < policy_size:
            raise ReplayError(
                f"Move {move_number} has out-of-range action {move.action_id}."
            )

        if index in selected:
            row = index - selected.start
            _write_policy(policy[row], move, set(env.legal_actions()), move_number)
            features[row] = env.features()
            value[row, 0] = record.black_result if player == 1 else -record.black_result

        if not env.act(move.action_id):
            raise ReplayError(
                f"Move {move_number} has illegal action {move.action_id}."
            )

    # Only an explicitly finished record can end by resignation.
    is_resignation = not env.is_terminal()
    if is_resignation and record.is_complete is not True:
        raise ReplayError("A nonterminal game requires an explicit completion marker.")
    score = env.eval_score(is_resign=is_resignation)
    if score != record.black_result:
        raise ReplayError(
            f"Replayed black result {score:g} disagrees with RE={record.black_result}."
        )

    return TrainingSamples(features=features, policy=policy, value=value)
