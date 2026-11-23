import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from .config_io import load_training_config, save_training_config

_REPO = Path(__file__).resolve().parents[3]
_TRAINING_CONFIGS = _REPO / "configs/training"
_SMOKE_CONFIG = _TRAINING_CONFIGS / "go_9x9_smoke.toml"


def test_snapshot_round_trip_uses_repository_paths(tmp_path, monkeypatch):
    pytest.importorskip("tomli_w")
    config = load_training_config(_SMOKE_CONFIG)
    config = replace(
        config,
        learner=replace(config.learner, device=torch.device("cpu"), seed=None),
        selfplay=replace(config.selfplay, initial_model=_REPO / "models/initial.pt"),
    )
    monkeypatch.chdir(tmp_path)
    save_training_config(config, "snapshot.toml")
    reloaded = load_training_config("snapshot.toml")
    assert reloaded == replace(config, learner=replace(config.learner, device="cpu"))
    assert reloaded.model.board_size == reloaded.learner.board_size == 9
    assert reloaded.selfplay.config_path == _REPO / "configs/minizero/go_9x9_az.cfg"
    assert "seed" not in tomllib.loads(Path("snapshot.toml").read_text(encoding="utf-8"))["learner"]


@pytest.mark.parametrize("name", ["go_9x9_smoke.toml", "go_9x9_az.toml"])
def test_existing_presets_use_one_mini_batch_and_ema_is_off(name):
    learner = load_training_config(_TRAINING_CONFIGS / name).learner
    assert learner.mini_batch_size is None
    assert not learner.ema_enabled


def test_local_ema_preset_round_trip(tmp_path):
    pytest.importorskip("tomli_w")
    config = load_training_config(_TRAINING_CONFIGS / "go_9x9_local_ema.toml")
    learner = config.learner
    schedule = config.schedule
    assert config.pool_id == "train-go-9x9-local-ema"
    assert (learner.batch_size, learner.mini_batch_size) == (1024, 512)
    assert (learner.min_train_samples, learner.target_samples) == (2000, 500_000)
    assert learner.ema_enabled and learner.ema_tau == 0.05
    assert (schedule.games_per_round, schedule.min_new_games_before_train) == (40, 40)
    assert schedule.train_steps_per_round == 4
    assert config.alphazero.num_simulations == 256
    assert not config.selfplay.quiet

    snapshot = tmp_path / "snapshot.toml"
    save_training_config(config, snapshot)
    assert load_training_config(snapshot) == config


@pytest.mark.parametrize(
    ("original", "changed", "field"),
    [
        ("[go]", "[extra]", "[extra]"),
        ("board_size = 9", "board_size = true", "[go].board_size"),
        ("pool_id = \"smoke-go-9x9\"", "pool_id = \"  \"", "[run].pool_id"),
        ("[model]", "[model]\nboard_size = 13", "[model].board_size"),
        ("in_channels = 18", "in_channels = 17", "[model].in_channels"),
        ("[learner]", "[learner]\nmini_batch_size = true", "[learner].mini_batch_size"),
        (
            "[learner]",
            "[learner]\nmini_batch_size = 3",
            "mini_batch_size must divide batch_size",
        ),
        ("[learner]", "[learner]\nema_tau = inf", "ema_tau must be in (0, 1]"),
        ("lr = 0.001\n", "", "[optimizer].lr"),
        ("num_simulations = 2", "num_simulation = 2", "[alphazero].num_simulation"),
        ("temperature = 1.0", "temperature = nan", "temperature must be finite"),
        ("[schedule]", "[schedule]\nmax_steps = 2", "[schedule].max_steps"),
        ("max_rounds = 2", "max_rounds = 0", "max_rounds must be an integer"),
    ],
)
def test_invalid_toml_identifies_the_field(tmp_path, original, changed, field):
    path = tmp_path / "invalid.toml"
    path.write_text(_SMOKE_CONFIG.read_text(encoding="utf-8").replace(original, changed, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(field)):
        load_training_config(path)
