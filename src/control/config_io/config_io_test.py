import re
from pathlib import Path

import pytest

from .config_io import load_training_config

_REPO = Path(__file__).resolve().parents[3]
_RUNS = _REPO / "runs"
_SMOKE_CONFIG = _RUNS / "smoke/config.toml"


def test_paths_follow_the_experiment_directory(tmp_path, monkeypatch):
    experiment = tmp_path / "elsewhere" / "experiment"
    experiment.mkdir(parents=True)
    toml = _SMOKE_CONFIG.read_text(encoding="utf-8").replace(
        "[selfplay]", '[selfplay]\ninitial_model = "models/initial.pt"'
    )
    path = experiment / "config.toml"
    path.write_text(toml, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = load_training_config(path.relative_to(tmp_path))
    assert config.model.board_size == config.learner.board_size == 9
    assert config.selfplay.engine_path == _REPO / "build/minizero/go/minizero_go"
    assert config.selfplay.config_path == experiment / "minizero_base.cfg"
    assert config.selfplay.initial_model == experiment / "models/initial.pt"


@pytest.mark.parametrize("name", ["smoke", "go-9x9-az"])
def test_existing_presets_use_one_mini_batch_and_ema_is_off(name):
    config = load_training_config(_RUNS / name / "config.toml")
    assert config.selfplay.config_path.is_file()
    assert config.learner.mini_batch_size is None
    assert not config.learner.ema_enabled
    assert config.schedule.checkpoint_every_rounds == 1


@pytest.mark.parametrize(
    ("original", "changed", "field"),
    [
        ("[go]", "[extra]", "[extra]"),
        ("board_size = 9", "board_size = true", "[go].board_size"),
        ("[go]", "[run]\npool_id = \"old-pool\"\n[go]", "[run]"),
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
        ("[schedule]", "[schedule]\ncheckpoint_every_rounds = 0", "checkpoint_every_rounds must be an integer"),
        ("[schedule]", "[schedule]\ncheckpoint_every_rounds = true", "[schedule].checkpoint_every_rounds"),
    ],
)
def test_invalid_toml_identifies_the_field(tmp_path, original, changed, field):
    path = tmp_path / "invalid.toml"
    path.write_text(_SMOKE_CONFIG.read_text(encoding="utf-8").replace(original, changed, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(field)):
        load_training_config(path)
