import hashlib
import json
import signal
import sys
import tomllib
from collections import deque
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch

from ...data.minizero_replay.minizero_replay import ReplayError
from ...data_types.selfplay_result.selfplay_result import SelfPlayResult
from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame
from ...data_types.training_samples.training_samples import TrainingSamples
from . import training_controller as controller

_ROOT = Path(__file__).resolve().parents[3]
_RULES = "env_board_size=9\nenv_go_komi=7\nenv_go_ko_rule=positional\n"
_MALFORMED = "SelfPlay true invalid record #"


class _Database:
    def __init__(self):
        self.games = []
        self.batches = deque()
        self.models = []
        self.after_initial_read = None
        self.after_batch = None
        self.other_source = SourceRun(uuid4(), "a" * 64, "b" * 40, _RULES)

    def append(self, source, text):
        game = StoredGame(uuid4(), len(self.games) + 1, text, source)
        self.games.append(game)
        return game

    def fetch_games_before(self, before_seq, limit):
        page = [
            game for game in reversed(self.games)
            if before_seq is None or game.ingest_seq < before_seq
        ][:limit]
        if before_seq is None and self.after_initial_read is not None:
            self.after_initial_read()
            self.after_initial_read = None
        return page

    def fetch_games_after(self, after_seq, limit):
        return [game for game in self.games if game.ingest_seq > after_seq][:limit]

    def collect(
        self, store, *, games, config_path, model_path,
        on_source_registered, on_game_committed, **_,
    ):
        assert store is self
        assert config_path.name == "minizero_base.cfg"
        records = self.batches.popleft()
        assert len(records) == games
        weights = model_path.read_bytes()
        self.models.append((model_path, weights))
        source = SourceRun(
            uuid4(), hashlib.sha256(weights).hexdigest(), "b" * 40, _RULES
        )
        on_source_registered(source)
        for record in records:
            game = self.append(source, record)
            on_game_committed(game.game_id, game.ingest_seq)
        if self.after_batch is not None:
            callback, self.after_batch = self.after_batch, None
            callback()
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
    pytest.importorskip("tensorboard")
    monkeypatch.delenv("XIAQI_EXPERIMENT_DIR", raising=False)
    directory = tmp_path / "experiment"
    directory.mkdir()
    config_path = directory / "config.toml"
    copyfile(_ROOT / "runs/smoke/config.toml", config_path)
    copyfile(_ROOT / "runs/smoke/minizero_base.cfg", directory / "minizero_base.cfg")
    db = _Database()
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(controller, "_SELFPLAY_MODEL_ROOT", shm)
    monkeypatch.setattr(controller, "_PAGE_SIZE", 2)
    monkeypatch.setattr(controller, "PostgreSQLStore", lambda connection: connection)
    monkeypatch.setattr(controller, "collect_live", db.collect)
    monkeypatch.setattr(controller, "replay_record", _replay)
    return db, config_path


def _configure(path, **changes):
    tables = tomllib.loads(path.read_text(encoding="utf-8"))
    for name, values in changes.items():
        tables[name].update(values)
    lines = []
    for name, table in tables.items():
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in table.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _events(directory, name):
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [event for event in events if event["event"] == name]


def _tensorboard(directory, axis="train"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(str(directory / "tensorboard" / axis)).Reload()


def test_two_rounds_use_exported_weights_and_resume(training):
    db, config_path = training
    db.batches.extend([[_record(1), _record(-1)], [_record(0), _record(1)]])
    original_config = config_path.read_text(encoding="utf-8")
    first = controller.TrainingController(config_path).run(db)
    assert first == config_path.parent
    checkpoints = first / "checkpoints"
    (first_path, first_weights), (second_path, second_weights) = db.models
    assert first_path == second_path
    assert first_path.name == "model.pt"
    assert first_path.parent.parent == controller._SELFPLAY_MODEL_ROOT
    assert not first_path.parent.exists()
    assert not (first / "selfplay_models").exists()
    assert first_weights != second_weights
    second_source = _events(first, "selfplay_started")[1]
    assert second_source["model_sha256"] == hashlib.sha256(second_weights).hexdigest()
    assert [event["step"] for event in _events(first, "model_exported")] == [0, 2]
    assert [event["step"] for event in _events(first, "checkpoint_saved")] == [2, 4]
    assert [event["step"] for event in _events(first, "round_finished")] == [2, 4]
    logs = _tensorboard(first)
    assert [item.step for item in logs.Scalars("train/total_loss")] == [1, 2, 3, 4]
    assert [item.step for item in _tensorboard(first, "rounds").Scalars("selfplay/black_win_rate")] == [1, 2]

    assert config_path.read_text(encoding="utf-8") == original_config
    started = _events(first, "run_started")[0]
    assert started["config_toml"] == original_config
    assert started["minizero_config"] == (first / "minizero_base.cfg").read_bytes().decode("utf-8")
    revised_config = original_config.replace("max_rounds = 2", "max_rounds = 1").replace(
        "warmup_train_steps = 0", "warmup_train_steps = 1"
    )
    config_path.write_text(revised_config, encoding="utf-8")
    with (first / "events.jsonl").open("ab") as history:
        history.write(b'{"event":')
    db.batches.append([_record(-1), _record(1)])
    resume_path = checkpoints / "step-000004.pt"
    resumed = controller.TrainingController(config_path).run(db, resume=True)
    assert resumed == first
    assert _events(resumed, "pool_restored")[-1]["sample_count"] == 8
    assert _events(resumed, "selfplay_started")[-1]["step"] == 5
    assert _events(resumed, "selfplay_started")[-1]["round"] == 1
    assert [event["step"] for event in _events(resumed, "checkpoint_saved")] == [2, 4, 5, 7]
    assert [item.step for item in _tensorboard(resumed).Scalars("train/total_loss")] == list(range(1, 8))
    assert [item.step for item in _tensorboard(resumed, "rounds").Scalars("selfplay/black_win_rate")] == [1, 2, 3]
    metadata = _events(resumed, "run_started")[-1]
    assert metadata["run_id"] != started["run_id"]
    assert metadata["config_toml"] == revised_config
    assert metadata["resume"] == str(resume_path)
    assert metadata["initial_weight_source"] == "warmup"


def test_wrong_experiment_is_rejected_before_connecting(training, monkeypatch, capsys):
    from scripts import train

    _, config_path = training
    monkeypatch.setenv("XIAQI_EXPERIMENT_DIR", str(config_path.parent.parent))
    monkeypatch.setenv("XIAQI_DATABASE_URL", "test")
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(Error=RuntimeError))
    assert train.main(["--config", str(config_path)]) == 1
    assert "outside the database experiment" in capsys.readouterr().err
    assert not (config_path.parent / "events.jsonl").exists()


def test_checkpoint_selection_precedes_events_and_only_accepts_latest(training):
    db, config_path = training
    trainer = controller.TrainingController(config_path)
    checkpoints = config_path.parent / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step-backup.pt").touch()
    with pytest.raises(FileNotFoundError, match="No training checkpoint"):
        trainer.run(db, resume=True)
    for name in ("step-9.pt", "step-10.pt"):
        (checkpoints / name).touch()
    assert controller._latest_checkpoint(checkpoints) == checkpoints / "step-10.pt"
    with pytest.raises(TypeError, match="boolean"):
        trainer.run(db, resume=checkpoints / "step-9.pt")
    with pytest.raises(ValueError, match="already has checkpoints"):
        trainer.run(db)
    assert not (config_path.parent / "events.jsonl").exists()


def test_another_trainer_cannot_enter_the_experiment(training):
    db, config_path = training
    _configure(config_path, schedule={"max_rounds": 1})
    db.batches.append([_record(1), _record(-1)])

    def try_second_trainer(event):
        if event["event"] == "run_started":
            with pytest.raises(controller.TrainingRunError, match="Another trainer"):
                controller.TrainingController(config_path).run(db)

    directory = controller.TrainingController(config_path, on_event=try_second_trainer).run(db)
    assert len(_events(directory, "run_started")) == 1
    db.batches.append([_record(1), _record(-1)])
    controller.TrainingController(config_path).run(db, resume=True)
    assert [event["step"] for event in _events(directory, "checkpoint_saved")] == [2, 4]


def test_cumulative_gate_and_win_rates_use_only_this_runs_games(training):
    db, config_path = training
    _configure(config_path, schedule={"games_per_round": 3, "max_rounds": 3})
    db.append(db.other_source, _record(1, 1))
    db.after_initial_read = lambda: db.append(db.other_source, _record(0))
    db.after_batch = lambda: db.append(db.other_source, _MALFORMED)
    db.batches.extend([
        [_record(1, 1), _record(-1, 0), _MALFORMED],
        [_record(0), _record(1, komi=6.5), _record(-1, model="invalid")],
        [_MALFORMED, _MALFORMED, _MALFORMED],
    ])

    directory = controller.TrainingController(config_path).run(db)
    first, second, third = _events(directory, "round_finished")
    assert (first["train_steps"], second["train_steps"], third["train_steps"]) == (0, 2, 0)
    assert [event["step"] for event in _events(directory, "checkpoint_saved")] == [2]
    assert db.models[0] == db.models[1]
    assert (first["games_scored"], first["games_with_samples"], first["sample_count"]) == (2, 1, 4)
    assert (first["black_win_rate"], first["white_win_rate"], first["draw_rate"]) == (0.5, 0.5, 0.0)
    assert second["games_scored"] == second["draw_rate"] == 1
    assert third["games_scored"] == 0 and "black_win_rate" not in third
    skipped = _events(directory, "game_skipped")
    assert [event["round"] for event in skipped if event["source_run_id"] == str(db.other_source.source_run_id)] == [2]
    logs = _tensorboard(directory, "rounds")
    assert [item.step for item in logs.Scalars("selfplay/black_win_rate")] == [1, 2]
    summary = _events(directory, "run_finished")[0]
    assert summary["round"] is None and summary["source_run_id"] is None
    assert summary["games_committed"] == 9
    assert summary["games_scored"] == 3
    for tag in ("black_win_rate", "white_win_rate", "draw_rate"):
        assert summary[tag] == pytest.approx(1 / 3)


def test_checkpoint_interval_counts_skipped_rounds_and_saves_final_update(training):
    db, config_path = training
    _configure(config_path, schedule={"max_rounds": 3, "checkpoint_every_rounds": 2})
    db.batches.extend([
        [_record(1), _record(-1)],
        [_MALFORMED, _MALFORMED],
        [_record(-1), _record(1)],
    ])
    directory = controller.TrainingController(config_path).run(db)

    assert [event["train_steps"] for event in _events(directory, "round_finished")] == [2, 0, 2]
    saved = _events(directory, "checkpoint_saved")
    assert [(event["round"], event["step"], event["reason"]) for event in saved] == [
        (2, 2, "periodic"),
        (3, 4, "final"),
    ]
    assert sorted(path.name for path in (directory / "checkpoints").glob("*.pt")) == [
        "step-000002.pt", "step-000004.pt",
    ]


def test_exports_ema_parameters_for_the_next_round(training):
    db, config_path = training
    _configure(config_path, learner={"mini_batch_size": 1, "ema_enabled": True})
    db.batches.extend([[_record(1), _record(-1)], [_record(0), _record(1)]])
    directory = controller.TrainingController(config_path).run(db)
    checkpoint = directory / "checkpoints/step-000002.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _, weights = db.models[1]
    exported = torch.jit.load(BytesIO(weights), map_location="cpu").model

    parameters = dict(exported.named_parameters())
    torch.testing.assert_close(parameters, state["ema_model"], rtol=0, atol=0)
    assert any(
        not torch.equal(weight, state["model"][name])
        for name, weight in parameters.items()
    )
    assert (
        _events(directory, "selfplay_started")[1]["model_sha256"]
        == hashlib.sha256(weights).hexdigest()
    )


def test_external_initial_model_is_pinned_in_snapshot(training, tmp_path):
    db, config_path = training
    external_model = tmp_path / "external.pt"
    external_model.write_bytes(b"external test weights")
    _configure(
        config_path,
        selfplay={"initial_model": "../external.pt"},
        schedule={"max_rounds": 1, "min_new_games_before_train": 3},
    )
    db.batches.append([_record(1), _record(-1)])
    directory = controller.TrainingController(config_path).run(db)
    frozen_model = directory / "selfplay_models/initial.pt"
    assert frozen_model.read_bytes() == external_model.read_bytes()
    model_path, weights = db.models[0]
    assert model_path.name == "model.pt" and not model_path.exists()
    assert weights == frozen_model.read_bytes()
    metadata = _events(directory, "run_started")[0]
    assert metadata["initial_model"] == str(external_model)
    assert metadata["pinned_initial_model"] == str(frozen_model)
    assert metadata["initial_weight_source"] == "initial_model"


def test_warmup_requires_ready_pool(training):
    db, config_path = training
    _configure(config_path, schedule={"warmup_train_steps": 1})
    with pytest.raises(controller.TrainingRunError, match="warmup: Warmup has 0 samples"):
        controller.TrainingController(config_path).run(db)
    directory = config_path.parent
    assert _events(directory, "run_failed")[0]["phase"] == "warmup"
    assert not _events(directory, "selfplay_started")


def test_export_failure_keeps_the_checkpoint(training, monkeypatch):
    db, config_path = training
    db.batches.append([_record(1), _record(-1)])
    export_model = controller.export_minizero_torchscript

    def export(model, path):
        if path.exists():
            raise OSError("Export test failure.")
        export_model(model, path)

    monkeypatch.setattr(controller, "export_minizero_torchscript", export)
    with pytest.raises(controller.TrainingRunError, match="export_model: Export test failure"):
        controller.TrainingController(config_path).run(db)
    directory = config_path.parent
    checkpoint = directory / "checkpoints/step-000002.pt"
    assert torch.load(checkpoint, map_location="cpu", weights_only=True)["step"] == 2
    assert _events(directory, "checkpoint_saved")[0]["path"] == str(checkpoint)
    assert _events(directory, "run_failed")[0]["phase"] == "export_model"
    assert not db.models[0][0].parent.exists()


def test_environment_failure_keeps_cursor_at_last_processed_game(training):
    db, config_path = training
    db.batches.append([_record(1), _record(-1, model="missing_environment")])
    with pytest.raises(controller.TrainingRunError, match="refresh_pool: MiniZero environment unavailable"):
        controller.TrainingController(config_path).run(db)
    directory = config_path.parent
    error = _events(directory, "run_failed")[0]
    assert error["ingest_seq"] == 1
    assert error["last_ingest_seq"] == 2
    assert not _events(directory, "game_skipped")


def test_cli_interrupt_while_collecting_saves_step_zero(training, monkeypatch, capsys):
    from scripts import train

    db, config_path = training
    db.batches.append([_record(1), _record(-1)])

    def interrupt():
        raise KeyboardInterrupt

    db.after_batch = interrupt
    monkeypatch.setenv("XIAQI_DATABASE_URL", "test")
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(
        Error=RuntimeError, connect=lambda *args, **kwargs: nullcontext(db),
    ))

    assert train.main(["--config", str(config_path)]) == 130
    directory = config_path.parent
    checkpoint = directory / "checkpoints/step-000000.pt"
    assert torch.load(checkpoint, map_location="cpu", weights_only=True)["step"] == 0
    error_output = capsys.readouterr().err
    assert str(checkpoint) in error_output and "-Resume" in error_output
    stopped = _events(directory, "run_interrupted")[0]
    assert stopped["phase"] == "selfplay"
    assert stopped["games_committed"] == stopped["last_ingest_seq"] == 2
    assert stopped["checkpoint_path"] == str(checkpoint)
    assert stopped["checkpoint_error"] is None
    assert not db.models[0][0].parent.exists()


@pytest.mark.parametrize("save_fails", [False, True])
def test_sigint_finishes_step_and_preserves_a_resumable_checkpoint(training, monkeypatch, save_fails):
    db, config_path = training
    _configure(config_path, learner={"ema_enabled": True})
    db.batches.extend([[_record(1), _record(-1)], [_record(0), _record(1)]])
    original_step = controller.Learner.step
    original_save = controller.Learner.save_checkpoint

    def step(learner):
        if learner.step_count == 2:
            signal.raise_signal(signal.SIGINT)
        return original_step(learner)

    def save(learner, path):
        if save_fails and learner.step_count == 3:
            raise OSError("Cannot write checkpoint.")
        original_save(learner, path)

    previous_handler = signal.getsignal(signal.SIGINT)
    with monkeypatch.context() as patch:
        patch.setattr(controller.Learner, "step", step)
        patch.setattr(controller.Learner, "save_checkpoint", save)
        with pytest.raises(controller.TrainingInterrupted) as caught:
            controller.TrainingController(config_path).run(db)

    directory = config_path.parent
    stopped = _events(directory, "run_interrupted")[0]
    assert stopped["phase"] == "train" and stopped["step"] == 3
    assert [item.step for item in _tensorboard(directory).Scalars("train/total_loss")] == [1, 2, 3]
    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert not db.models[0][0].parent.exists()

    step_to_resume = 2 if save_fails else 3
    checkpoint = directory / "checkpoints" / f"step-{step_to_resume:06d}.pt"
    assert caught.value.checkpoint_path == checkpoint
    assert stopped["checkpoint_path"] == str(checkpoint)
    assert stopped["checkpoint_error"] == ("Cannot write checkpoint." if save_fails else None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == step_to_resume and "ema_model" in state
    assert {entry["step"].item() for entry in state["optimizer"]["state"].values()} == {step_to_resume}

    _configure(config_path, schedule={"max_rounds": 1})
    db.batches.append([_record(1), _record(-1)])
    resumed = controller.TrainingController(config_path).run(db, resume=True)
    assert resumed == directory
    assert _events(resumed, "round_finished")[-1]["step"] == step_to_resume + 2
    assert _events(resumed, "checkpoint_saved")[-1]["round"] == 1
    assert [item.step for item in _tensorboard(resumed).Scalars("train/total_loss")] == list(
        range(1, step_to_resume + 3)
    )


def test_sigint_during_checkpoint_reuses_the_completed_write(training, monkeypatch):
    db, config_path = training
    db.batches.append([_record(1), _record(-1)])
    original_save = controller.Learner.save_checkpoint
    writes = []

    def save(learner, path):
        writes.append(path)
        original_save(learner, path)
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(controller.Learner, "save_checkpoint", save)
    with pytest.raises(controller.TrainingInterrupted) as caught:
        controller.TrainingController(config_path).run(db)

    directory = config_path.parent
    checkpoint = directory / "checkpoints/step-000002.pt"
    assert writes == [checkpoint]
    assert caught.value.checkpoint_path == checkpoint
    assert _events(directory, "run_interrupted")[0]["phase"] == "save_checkpoint"
