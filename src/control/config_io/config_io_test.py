import re
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from .config_io import load_training_config, save_training_config

_REPO = Path(__file__).resolve().parents[3]
_SMOKE_CONFIG = _REPO / "configs/training/go_9x9_smoke.toml"


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


@pytest.mark.parametrize(
    ("original", "changed", "field"),
    [
        ("[go]", "[extra]", "[extra]"),
        ("board_size = 9", "board_size = true", "[go].board_size"),
        ("pool_id = \"smoke-go-9x9\"", "pool_id = \"  \"", "[run].pool_id"),
        ("[model]", "[model]\nboard_size = 13", "[model].board_size"),
        ("in_channels = 18", "in_channels = 17", "[model].in_channels"),
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
