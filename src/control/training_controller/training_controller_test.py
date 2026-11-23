import hashlib
import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from ...data.minizero_replay.minizero_replay import ReplayError
from ...data_types.selfplay_result.selfplay_result import SelfPlayResult
from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame
from ...data_types.training_samples.training_samples import TrainingSamples
from ..config_io.config_io import load_training_config
from . import training_controller as controller

_ROOT = Path(__file__).resolve().parents[3]
_RULES = "env_board_size=9\nenv_go_komi=7\nenv_go_ko_rule=positional\n"
_MALFORMED = "SelfPlay true invalid record #"


class _Database:
    def __init__(self, pool_id):
        self.pool_id = pool_id
        self.games = []
        self.batches = deque()
        self.after_initial_read = None
        self.after_batch = None
        self.other_source = SourceRun(uuid4(), pool_id, "a" * 64, "b" * 40, _RULES)

    def append(self, source, text):
        game = StoredGame(uuid4(), len(self.games) + 1, text, source)
        self.games.append(game)
        return game

    def fetch_games_before(self, pool_id, before_seq, limit):
        assert pool_id == self.pool_id
        page = [
            game for game in reversed(self.games)
            if before_seq is None or game.ingest_seq < before_seq
        ][:limit]
        if before_seq is None and self.after_initial_read is not None:
            self.after_initial_read()
            self.after_initial_read = None
        return page

    def fetch_games_after(self, pool_id, after_seq, limit):
        assert pool_id == self.pool_id
        return [game for game in self.games if game.ingest_seq > after_seq][:limit]

    def collect(
        self, store, *, pool_id, games, config_path, model_path,
        on_source_registered, on_game_committed, **_,
    ):
        assert store is self and pool_id == self.pool_id
        assert config_path.name == "minizero_base.cfg"
        records = self.batches.popleft()
        assert len(records) == games
        source = SourceRun(
            uuid4(), pool_id, hashlib.sha256(model_path.read_bytes()).hexdigest(), "b" * 40, _RULES
        )
        on_source_registered(source)
        for record in records:
            game = self.append(source, record)
            on_game_committed(game.game_id, game.ingest_seq)
        if self.after_batch is not None:
            self.after_batch()
            self.after_batch = None
        return SelfPlayResult(games, game.ingest_seq, source)


def _record(result, positions=2, *, komi=7, model="test"):
    moves = (";B[1]P[1:1]", ";W[2]P[2:1]")
    return (
        f"SelfPlay true {positions} {positions} {result} "
        f"(;GM[go_9x9]SZ[9]KM[{komi}]RE[{result}]DLEN[0-{positions - 1}]EV[{model}]"
        + "".join(moves[:positions]) + ") #"
    )


def _replay(record, *, ko_rule):
    assert ko_rule == "positional"
    if record.model_name == "invalid":
        raise ReplayError("Invalid move in the test record.")
    if record.model_name == "missing_environment":
        raise RuntimeError("MiniZero environment unavailable.")
    count = len(record.sample_range)
    policy = np.zeros((count, 82), dtype=np.float32)
    policy[:, 1] = 1
    return TrainingSamples(
        np.zeros((count, 18, 9, 9), dtype=np.float32),
        policy,
        np.full((count, 1), record.black_result, dtype=np.float32),
    )


@pytest.fixture
def training(tmp_path, monkeypatch):
    pytest.importorskip("tomli_w")
    pytest.importorskip("tensorboard")
    config = load_training_config(_ROOT / "configs/training/go_9x9_smoke.toml")
    config = replace(config, runs_dir=tmp_path / "runs")
    db = _Database(config.pool_id)
    monkeypatch.setattr(controller, "_PAGE_SIZE", 2)
    monkeypatch.setattr(controller, "PostgreSQLReader", lambda connection: connection)
    monkeypatch.setattr(controller, "PostgreSQLStore", lambda connection: connection)
    monkeypatch.setattr(controller, "collect_live", db.collect)
    monkeypatch.setattr(controller, "replay_record", _replay)
    return config, db


def _events(directory, name):
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [event for event in events if event["event"] == name]


def _tensorboard(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(str(directory / "tensorboard")).Reload()


def test_two_rounds_use_exported_weights_and_resume(training):
    config, db = training
    db.batches.extend([[_record(1), _record(-1)], [_record(0), _record(1)]])
    first = controller.TrainingController(config, db).run()
    checkpoints = first / "checkpoints"
    model = first / "selfplay_models" / "step-000002.pt"
    second_source = _events(first, "selfplay_started")[1]
    assert second_source["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert [event["step"] for event in _events(first, "round_finished")] == [2, 4]
    logs = _tensorboard(first)
    assert [item.step for item in logs.Scalars("train/total_loss")] == [1, 2, 3, 4]
    assert [item.step for item in logs.Scalars("selfplay/black_win_rate")] == [1, 2]

    snapshot = load_training_config(first / "config.toml")
    assert snapshot.selfplay.config_path == first / "minizero_base.cfg"
    assert snapshot.selfplay.config_path.read_bytes() == config.selfplay.config_path.read_bytes()
    snapshot = replace(snapshot, schedule=replace(snapshot.schedule, max_rounds=1, warmup_train_steps=1))
    db.batches.append([_record(-1), _record(1)])
    resume_path = checkpoints / "step-000004.pt"
    resumed = controller.TrainingController(snapshot, db).run(resume=resume_path)
    assert resumed != first
    assert _events(resumed, "pool_restored")[0]["sample_count"] == 8
    assert _events(resumed, "selfplay_started")[0]["step"] == 5
    assert _events(resumed, "selfplay_started")[0]["round"] == 1
    assert [item.step for item in _tensorboard(resumed).Scalars("train/total_loss")] == [5, 6, 7]
    metadata = json.loads((resumed / "run.json").read_text(encoding="utf-8"))
    assert metadata["resume"] == str(resume_path)
    assert metadata["initial_weight_source"] == "warmup"


def test_cumulative_gate_and_win_rates_use_only_this_runs_games(training):
    config, db = training
    config = replace(config, schedule=replace(config.schedule, games_per_round=3, max_rounds=3))
    db.append(db.other_source, _record(1, 1))
    db.after_initial_read = lambda: db.append(db.other_source, _record(0))
    db.after_batch = lambda: db.append(db.other_source, _MALFORMED)
    db.batches.extend([
        [_record(1, 1), _record(-1, 0), _MALFORMED],
        [_record(0), _record(1, komi=6.5), _record(-1, model="invalid")],
        [_MALFORMED, _MALFORMED, _MALFORMED],
    ])

    directory = controller.TrainingController(config, db).run()
    first, second, third = _events(directory, "round_finished")
    assert (first["train_steps"], second["train_steps"], third["train_steps"]) == (0, 2, 0)
    assert (first["games_scored"], first["games_with_samples"], first["sample_count"]) == (2, 1, 4)
    assert (first["black_win_rate"], first["white_win_rate"], first["draw_rate"]) == (0.5, 0.5, 0.0)
    assert second["games_scored"] == second["draw_rate"] == 1
    assert third["games_scored"] == 0 and "black_win_rate" not in third
    skipped = _events(directory, "game_skipped")
    assert [event["round"] for event in skipped if event["source_run_id"] == str(db.other_source.source_run_id)] == [2]
    logs = _tensorboard(directory)
    assert [item.step for item in logs.Scalars("selfplay/black_win_rate")] == [1, 2]
    summary = _events(directory, "run_finished")[0]
    assert summary["round"] is None and summary["source_run_id"] is None
    assert summary["games_committed"] == 9
    assert summary["games_scored"] == 3
    for tag in ("black_win_rate", "white_win_rate", "draw_rate"):
        assert summary[tag] == pytest.approx(1 / 3)


def test_exports_ema_parameters_for_the_next_round(training):
    config, db = training
    config = replace(
        config,
        learner=replace(config.learner, mini_batch_size=1, ema_enabled=True),
    )
    db.batches.extend([[_record(1), _record(-1)], [_record(0), _record(1)]])
    directory = controller.TrainingController(config, db).run()
    checkpoint = directory / "checkpoints/step-000002.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_path = directory / "selfplay_models/step-000002.pt"
    exported = torch.jit.load(str(model_path), map_location="cpu").model

    parameters = dict(exported.named_parameters())
    torch.testing.assert_close(parameters, state["ema_model"], rtol=0, atol=0)
    assert any(
        not torch.equal(weight, state["model"][name])
        for name, weight in parameters.items()
    )
    assert (
        _events(directory, "selfplay_started")[1]["model_sha256"]
        == hashlib.sha256(model_path.read_bytes()).hexdigest()
    )


def test_external_initial_model_is_pinned_in_snapshot(training, tmp_path):
    config, db = training
    external_model = tmp_path / "external.pt"
    external_model.write_bytes(b"external test weights")
    config = replace(
        config,
        selfplay=replace(config.selfplay, initial_model=external_model),
        schedule=replace(config.schedule, max_rounds=1, min_new_games_before_train=3),
    )
    db.batches.append([_record(1), _record(-1)])
    directory = controller.TrainingController(config, db).run()
    snapshot = load_training_config(directory / "config.toml")
    frozen_model = directory / "selfplay_models/initial.pt"
    assert snapshot.selfplay.initial_model == frozen_model
    assert frozen_model.read_bytes() == external_model.read_bytes()
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert metadata["initial_model"] == str(external_model)
    assert metadata["initial_weight_source"] == "initial_model"


def test_warmup_requires_ready_pool(training):
    config, db = training
    config = replace(config, schedule=replace(config.schedule, warmup_train_steps=1))
    with pytest.raises(controller.TrainingRunError, match="warmup: Warmup has 0 samples"):
        controller.TrainingController(config, db).run()
    directory = next(config.runs_dir.iterdir())
    assert _events(directory, "run_failed")[0]["phase"] == "warmup"
    assert not _events(directory, "selfplay_started")


def test_export_failure_keeps_the_checkpoint(training, monkeypatch):
    config, db = training
    db.batches.append([_record(1), _record(-1)])
    export_model = controller.export_minizero_torchscript

    def export(model, path):
        if path.name != "initial.pt":
            raise OSError("Export test failure.")
        export_model(model, path)

    monkeypatch.setattr(controller, "export_minizero_torchscript", export)
    with pytest.raises(controller.TrainingRunError, match="export_model: Export test failure"):
        controller.TrainingController(config, db).run()
    directory = next(config.runs_dir.iterdir())
    checkpoint = directory / "checkpoints/step-000002.pt"
    assert torch.load(checkpoint, map_location="cpu", weights_only=True)["step"] == 2
    assert _events(directory, "checkpoint_saved")[0]["path"] == str(checkpoint)
    assert _events(directory, "run_failed")[0]["phase"] == "export_model"


def test_environment_failure_keeps_cursor_at_last_processed_game(training):
    config, db = training
    db.batches.append([_record(1), _record(-1, model="missing_environment")])
    with pytest.raises(controller.TrainingRunError, match="refresh_pool: MiniZero environment unavailable"):
        controller.TrainingController(config, db).run()
    directory = next(config.runs_dir.iterdir())
    error = _events(directory, "run_failed")[0]
    assert error["ingest_seq"] == 1
    assert error["last_ingest_seq"] == 2
    assert not _events(directory, "game_skipped")
