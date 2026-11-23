"""采集或导入完整的 MiniZero 棋谱，并逐盘提交。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from ctypes import c_float
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

from ...data.minizero_selfplay.minizero_selfplay import (
    iter_complete_records,
    read_effective_config,
    selfplay_stdout,
)
from ...data_types.selfplay_result.selfplay_result import SelfPlayResult
from ...data_types.source_run.source_run import SourceRun

if TYPE_CHECKING:
    from ...data.postgresql_store.postgresql_store import PostgreSQLStore
    from ...data_types.training_config.training_config import TrainingConfig

_Setting = bool | int | float | str
_SourceCallback = Callable[[SourceRun], None]
_GameCallback = Callable[[UUID, int], None]


def training_overrides(config: TrainingConfig) -> dict[str, _Setting]:
    go, selfplay, az = config.go, config.selfplay, config.alphazero
    return {
        "env_board_size": go.board_size,
        "env_go_komi": go.komi,
        "env_go_ko_rule": go.ko_rule,
        "zero_num_threads": selfplay.num_threads,
        "zero_num_parallel_games": selfplay.num_parallel_games,
        "program_seed": selfplay.seed,
        "program_auto_seed": selfplay.auto_seed,
        "program_quiet": selfplay.quiet,
        "actor_num_simulation": az.num_simulations,
        "actor_mcts_puct_base": az.puct_base,
        "actor_mcts_puct_init": az.puct_init,
        "actor_mcts_reward_discount": az.reward_discount,
        "actor_mcts_value_rescale": az.value_rescale,
        "actor_select_action_by_count": az.action_selection == "max_count",
        "actor_select_action_by_softmax_count": az.action_selection == "softmax_count",
        "actor_select_action_softmax_temperature": az.temperature,
        "actor_use_dirichlet_noise": az.use_dirichlet_noise,
        "actor_dirichlet_noise_alpha": az.dirichlet_alpha,
        "actor_dirichlet_noise_epsilon": az.dirichlet_epsilon,
        "actor_use_random_rotation_features": az.random_rotation_features,
        "actor_resign_threshold": az.resign_threshold,
        "zero_disable_resign_ratio": az.disable_resign_ratio,
        "actor_use_gumbel": False,
        "actor_use_gumbel_noise": False,
        "zero_actor_intermediate_sequence_length": 0,
    }


def collect_live(
    store: PostgreSQLStore,
    *,
    pool_id: str,
    games: int,
    engine_path: Path,
    config_path: Path,
    model_path: Path,
    overrides: str | Mapping[str, _Setting] = "",
    on_source_registered: _SourceCallback | None = None,
    on_game_committed: _GameCallback | None = None,
) -> SelfPlayResult:
    """Collect a fresh batch with raw CLI or typed training overrides."""
    if type(games) is not int or games < 1:
        raise ValueError("games must be a positive integer.")

    engine = engine_path.resolve()
    commit_file = engine.with_name(f"{engine.name}.commit")
    if not commit_file.is_file():
        raise ValueError(f"Missing {commit_file}. Build the engine with scripts/build_minizero.sh.")
    model_name = str(model_path)
    if model_name != model_name.strip() or any(char in model_name for char in ":#\r\n"):
        raise ValueError("The model path cannot contain ':', '#', line breaks or surrounding whitespace.")

    expected = None
    if isinstance(overrides, str):
        conf_str = ":".join(filter(None, (overrides, f"nn_file_name={model_name}")))
    else:
        board_size = overrides.get("env_board_size")
        if type(board_size) is not int or not 1 <= board_size <= 19:
            raise ValueError("Typed overrides require env_board_size between 1 and 19.")
        _validate_model(model_path, board_size)
        expected = {**overrides, "nn_file_name": model_name}
        conf_str = ":".join(
            f"{name}={str(value).lower() if isinstance(value, bool) else value}"
            for name, value in expected.items()
        )

    with model_path.open("rb") as model_file:
        model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    command = [str(engine), "-conf_file", str(config_path), "-conf_str", conf_str]
    effective_config = read_effective_config(command)
    if expected is not None:
        _check_effective_config(effective_config, expected)

    source = SourceRun(
        source_run_id=uuid4(),
        pool_id=pool_id,
        model_sha256=model_sha256,
        minizero_commit=commit_file.read_text(encoding="ascii").strip(),
        effective_config=effective_config,
    )
    result = _ingest(
        store, source, lambda: selfplay_stdout(command), games,
        on_source_registered, on_game_committed,
    )
    if result.games_committed < games:
        raise RuntimeError(f"MiniZero ended after {result.games_committed} of {games} requested games.")
    return result


def import_records(
    store: PostgreSQLStore,
    source: SourceRun,
    record_file: Path,
    *,
    on_source_registered: _SourceCallback | None = None,
    on_game_committed: _GameCallback | None = None,
) -> SelfPlayResult:
    """Use the original source metadata and file order when retrying an import."""
    result = _ingest(
        store, source, lambda: record_file.open(encoding="utf-8"), None,
        on_source_registered, on_game_committed,
    )
    if not result.games_committed:
        raise ValueError(f"No complete SelfPlay records in {record_file}.")
    return result


def parse_effective_config(text: str) -> dict[str, str]:
    """Read the key/value lines emitted by MiniZero -gen."""
    options = {}
    for line in text.splitlines():
        key, separator, value = line.partition("#")[0].partition("=")
        if separator and key.strip():
            options[key.strip()] = value.strip()
    return options


def effective_float(value: float) -> float:
    # MiniZero stores these settings as C++ floats and prints six significant digits.
    return float(f"{c_float(value).value:.6g}")


def _ingest(
    store: PostgreSQLStore,
    source: SourceRun,
    open_lines: Callable[[], AbstractContextManager[Iterable[str]]],
    limit: int | None,
    on_source_registered: _SourceCallback | None,
    on_game_committed: _GameCallback | None,
) -> SelfPlayResult:
    store.register_source(source)
    if on_source_registered is not None:
        on_source_registered(source)

    count = last_ingest_seq = 0
    with open_lines() as lines:
        records = islice(iter_complete_records(lines), limit)
        for count, record in enumerate(records, start=1):
            game_id = uuid5(source.source_run_id, str(count))
            last_ingest_seq = store.append_game(source.pool_id, game_id, source.source_run_id, record)
            if on_game_committed is not None:
                on_game_committed(game_id, last_ingest_seq)
    return SelfPlayResult(count, last_ingest_seq, source)


def _validate_model(path: Path, board_size: int) -> None:
    import torch

    try:
        model = torch.jit.load(str(path), map_location="cpu")
        actual = (
            model.get_type_name(),
            model.get_input_channel_height(),
            model.get_input_channel_width(),
            model.get_num_input_channels(),
            model.get_action_size(),
        )
    except (AttributeError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot read MiniZero TorchScript model metadata from {path}: {error}") from error
    expected = ("alphazero", board_size, board_size, 18, board_size**2 + 1)
    if actual != expected:
        raise ValueError(
            f"Incompatible MiniZero model {path}. Expected {expected}, got {actual} "
            "(type, height, width, channels, actions)."
        )


def _check_effective_config(text: str, expected: Mapping[str, _Setting]) -> None:
    options = parse_effective_config(text)
    for name, value in expected.items():
        actual = options.get(name)
        if actual is None or not _option_matches(actual, value):
            raise ValueError(f"MiniZero effective {name} is {actual!r}, expected {value!r}.")


def _option_matches(actual: str, expected: _Setting) -> bool:
    try:
        if isinstance(expected, bool):
            return actual.lower() in (("true", "1") if expected else ("false", "0"))
        if isinstance(expected, int):
            return int(actual) == expected
        if isinstance(expected, float):
            return float(actual) == effective_float(expected)
    except ValueError:
        return False
    return actual == expected
