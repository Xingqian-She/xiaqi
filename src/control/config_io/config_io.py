"""读取、解析并校验实验的训练配置。"""

import tomllib
from dataclasses import MISSING, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

import torch

from ...data_types.alphazero_config.alphazero_config import AlphaZeroConfig
from ...data_types.go_rules_config.go_rules_config import GoRulesConfig
from ...data_types.learner_config.learner_config import LearnerConfig
from ...data_types.optimizer_config.optimizer_config import OptimizerConfig
from ...data_types.selfplay_config.selfplay_config import SelfPlayConfig
from ...data_types.training_config.training_config import TrainingConfig
from ...data_types.training_schedule.training_schedule import TrainingSchedule
from ...data_types.transformer_config.transformer_config import TransformerConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SECTIONS = ("go", "model", "learner", "optimizer", "selfplay", "alphazero", "schedule")
_Config = TypeVar("_Config")


def load_training_config(path: str | Path, *, text: str | None = None) -> TrainingConfig:
    path = Path(path).resolve()
    tables = tomllib.loads(path.read_text(encoding="utf-8") if text is None else text)
    unknown = tables.keys() - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown configuration table [{min(unknown)}].")

    go = _load_section(tables, "go", GoRulesConfig)
    config = TrainingConfig(
        go=go,
        model=_load_section(tables, "model", TransformerConfig, board_size=go.board_size),
        learner=_load_section(tables, "learner", LearnerConfig, board_size=go.board_size),
        optimizer=_load_section(tables, "optimizer", OptimizerConfig),
        selfplay=_load_section(tables, "selfplay", SelfPlayConfig),
        alphazero=_load_section(tables, "alphazero", AlphaZeroConfig),
        schedule=_load_section(tables, "schedule", TrainingSchedule),
    )
    validate_training_config(config)
    return resolve_training_paths(config, path.parent)


def resolve_training_paths(config: TrainingConfig, directory: Path) -> TrainingConfig:
    selfplay = config.selfplay
    return replace(
        config,
        selfplay=replace(
            selfplay,
            engine_path=(_REPO_ROOT / selfplay.engine_path).resolve(),
            config_path=(directory / selfplay.config_path).resolve(),
            initial_model=(directory / selfplay.initial_model).resolve()
            if selfplay.initial_model is not None
            else None,
        ),
    )


def validate_training_config(config: TrainingConfig) -> None:
    if config.go.board_size % 2 == 0:
        raise ValueError("[go].board_size must be odd for the Transformer.")
    for section in ("model", "learner"):
        if getattr(config, section).board_size != config.go.board_size:
            raise ValueError(f"[{section}].board_size must match [go].board_size.")

    model = config.model
    if model.in_channels != 18:
        raise ValueError("[model].in_channels must be 18 for MiniZero Go.")
    for name in ("model_dim", "num_heads", "ffn_hidden_dim", "num_layers", "value_hidden_dim"):
        if getattr(model, name) < 1:
            raise ValueError(f"[model].{name} must be positive.")
    if model.num_stem_blocks < 0:
        raise ValueError("[model].num_stem_blocks must be nonnegative.")
    if model.model_dim % (4 * model.num_heads) != 0:
        raise ValueError("[model].model_dim must be divisible by 4 * [model].num_heads.")

    learner = config.learner
    if learner.seed is not None and learner.seed < 0:
        raise ValueError("[learner].seed must be nonnegative.")
    try:
        torch.device(learner.device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"[learner].device is invalid: {error}") from error


def _table(tables: dict[str, Any], name: str) -> dict[str, Any]:
    table = tables.get(name, {})
    if not isinstance(table, dict):
        raise ValueError(f"[{name}] must be a table.")
    return table


def _check_fields(table: dict[str, Any], valid: set[str], name: str) -> None:
    unknown = table.keys() - valid
    if unknown:
        raise ValueError(f"Unknown configuration field [{name}].{min(unknown)}.")


def _load_section(
    tables: dict[str, Any],
    name: str,
    config_class: type[_Config],
    *,
    board_size: int | None = None,
) -> _Config:
    table = _table(tables, name)
    config_fields = {field.name: field for field in fields(config_class)}
    _check_fields(table, set(config_fields), name)
    values = {
        key: _read_value(value, config_fields[key].type, f"[{name}].{key}")
        for key, value in table.items()
    }
    if board_size is not None:
        values.setdefault("board_size", board_size)
    for key, field in config_fields.items():
        if key not in values and field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"[{name}].{key} is required.")
    try:
        return config_class(**values)
    except ValueError as error:
        raise ValueError(f"[{name}] {error}") from error


def _read_value(value: Any, annotation: Any, name: str) -> Any:
    choices = get_args(annotation)
    if get_origin(annotation) is Literal:
        if value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}.")
        return value
    if annotation is Path or Path in choices:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string.")
        return Path(value)
    if annotation is float:
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a number.")
        try:
            return float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite number.") from error
    expected = choices or (annotation,)
    if type(value) not in expected:
        raise ValueError(f"{name} must have type {expected[0].__name__}.")
    return value
