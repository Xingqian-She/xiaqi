"""串行运行 MiniZero 自对弈和模型更新。"""

from __future__ import annotations

import os
import re
import signal
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from ctypes import c_float
from dataclasses import replace
from functools import partial
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import current_thread, main_thread
from time import monotonic
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID

from filelock import FileLock, Timeout
from torch.optim import AdamW

from ...core.learner.learner import Learner
from ...core.model.transformer.transformer import Transformer
from ...data.minizero_record.minizero_record import RecordParseError, parse_record
from ...data.minizero_replay.minizero_replay import ReplayError, replay_record
from ...data.postgresql.postgresql import PostgreSQLStore
from ...data_types.game_record.game_record import GameRecord
from ...data_types.go_rules_config.go_rules_config import GoRulesConfig
from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame
from ...data_types.training_config.training_config import TrainingConfig
from ...integrations.minizero.torchscript_export import export_minizero_torchscript
from ..config_io.config_io import load_training_config
from ..run_log.run_log import EventCallback, RunLog
from ..selfplay_collector.selfplay_collector import (
    collect_live,
    effective_float,
    parse_effective_config,
    training_overrides,
)

if TYPE_CHECKING:
    from psycopg import Connection

_PAGE_SIZE = 256
_SELFPLAY_MODEL_ROOT = Path("/dev/shm")


class TrainingRunError(RuntimeError):
    """A training run failed in the phase named in the error message."""


class TrainingInterrupted(KeyboardInterrupt):
    """A training run was interrupted, with the last usable checkpoint if available."""

    def __init__(self, checkpoint_path: Path | None, checkpoint_error: str | None) -> None:
        super().__init__("Training interrupted.")
        self.checkpoint_path = checkpoint_path
        self.checkpoint_error = checkpoint_error


class TrainingController:
    """Use one caller-managed connection for serial collection and training."""

    def __init__(
        self,
        config_path: Path,
        *,
        on_event: EventCallback | None = None,
    ) -> None:
        self._config_path = config_path.resolve(strict=True)
        experiment = os.environ.get("XIAQI_EXPERIMENT_DIR")
        if experiment and self._config_path.parent != Path(experiment).resolve(strict=True):
            raise ValueError(
                f"Configuration {self._config_path} is outside the database experiment {experiment}."
            )
        self._on_event = on_event

    def run(self, connection: Connection, *, resume: bool = False) -> Path:
        """Train with exclusive access to this experiment and its latest checkpoint."""
        if type(resume) is not bool:
            raise TypeError("resume must be a boolean selecting the experiment's latest checkpoint.")
        directory = self._config_path.parent
        lock = FileLock(directory / ".training.lock")
        try:
            acquired = lock.acquire(timeout=0)
        except Timeout as error:
            raise TrainingRunError(f"Another trainer is using experiment {directory}.") from error
        with acquired:
            return self._run_locked(connection, resume)

    def _run_locked(self, connection: Connection, resume: bool) -> Path:
        self._log = RunLog(self._config_path.parent, self._on_event)
        checkpoint = _latest_checkpoint(self._log.checkpoints)
        if resume and checkpoint is None:
            raise FileNotFoundError(f"No training checkpoint found in {self._log.checkpoints}.")
        if not resume and checkpoint is not None:
            raise ValueError("The experiment already has checkpoints. Use --resume.")
        config_text = self._config_path.read_text(encoding="utf-8")
        self._config = load_training_config(self._config_path, text=config_text)
        self._store = PostgreSQLStore(connection)
        self._phase = "startup"
        self._round: int | None = None
        self._step = 0
        self._cursor = 0
        self._source_run_id: UUID | None = None
        self._games_committed = 0
        self._last_committed_seq = 0
        self._checkpoint_safe = False
        self._last_checkpoint_step = 0
        self._last_checkpoint_path: Path | None = None

        try:
            self._log.open()
            with TemporaryDirectory(prefix="xiaqi-selfplay-", dir=_SELFPLAY_MODEL_ROOT) as directory:
                summary = self._execute(checkpoint, config_text, Path(directory) / "model.pt")
            self._phase = "finish"
            self._log.close_metrics()
            self._event("run_finished", round=None, source_run_id=None, **summary)
            self._log.close()
        except KeyboardInterrupt as error:
            raise self._interrupt(error) from error
        except Exception as error:
            self._abort("run_failed", error)
            raise TrainingRunError(f"{self._phase}: {error}") from error
        return self._log.directory

    def _execute(
        self, resume: Path | None, config_toml: str, model_path: Path
    ) -> dict[str, int | float]:
        self._phase = "snapshot_config"
        config, external_model = self._snapshot_config(
            resume, config_toml, model_path.with_name("minizero_base.cfg")
        )

        self._phase = "initialize_learner"
        self._learner = Learner(
            lambda: Transformer.from_config(config.model),
            partial(AdamW, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay),
            config.learner,
        )
        if resume is not None:
            self._phase = "restore_checkpoint"
            resume = resume.resolve()
            self._learner.load_checkpoint(resume)
            self._step = self._learner.step_count
            if int(resume.stem.removeprefix("step-")) != self._step:
                raise ValueError(f"Checkpoint step {self._step} does not match {resume.name}.")
            self._last_checkpoint_step = self._step
            self._last_checkpoint_path = resume
            self._checkpoint_safe = True
            self._event("checkpoint_restored", path=resume)
        else:
            self._checkpoint_safe = True
        start_step = self._step
        self._log.open_metrics(start_step)

        self._phase = "backfill_pool"
        self._backfill_pool()
        status = self._learner.pool_status()
        self._event(
            "pool_restored",
            ingest_seq=self._cursor,
            sample_count=status.sample_count,
            game_count=status.game_count,
        )

        warmup_steps = config.schedule.warmup_train_steps
        if warmup_steps:
            self._phase = "warmup"
            if not status.ready:
                raise ValueError(
                    f"Warmup has {status.sample_count} samples. "
                    f"Training requires at least {config.learner.min_train_samples}."
                )
            self._train(warmup_steps)
            self._save_checkpoint("warmup")
        if external_model is not None:
            copyfile(external_model, model_path)
            self._event("initial_model_prepared", path=model_path)
        else:
            self._export_model(model_path)

        summary = self._run_rounds(config, model_path)
        return {**summary, "train_steps": self._step - start_step}

    def _snapshot_config(
        self, resume: Path | None, config_toml: str, base_config: Path
    ) -> tuple[TrainingConfig, Path | None]:
        config = self._config
        selfplay = config.selfplay
        external_model = None
        if config.schedule.warmup_train_steps:
            initial_weight_source = "warmup"
        elif resume is not None:
            initial_weight_source = "resume"
        elif selfplay.initial_model is not None:
            initial_weight_source = "initial_model"
            external_model = self._log.selfplay_models / "initial.pt"
        else:
            initial_weight_source = "new_model"

        base_contents = selfplay.config_path.read_bytes()
        base_config.write_bytes(base_contents)
        self._event(
            "run_started",
            run_dir=self._log.directory,
            config_toml=config_toml,
            minizero_base_source=selfplay.config_path,
            minizero_config=base_contents.decode("utf-8"),
            initial_weight_source=initial_weight_source,
            initial_model=selfplay.initial_model,
            pinned_initial_model=external_model,
            resume=resume.resolve() if resume is not None else None,
        )
        if external_model is not None and selfplay.initial_model is not None:
            external_model.parent.mkdir(exist_ok=True)
            if selfplay.initial_model != external_model:
                copyfile(selfplay.initial_model, external_model)
        return replace(config, selfplay=replace(selfplay, config_path=base_config)), external_model

    def _backfill_pool(self) -> None:
        page = self._store.fetch_games_before(None, _PAGE_SIZE)
        if page:
            self._cursor = page[0].ingest_seq
        while page:
            for game in page:
                self._replay_game(game, position="oldest")
                if self._learner.pool_status().sample_count >= self._config.learner.target_samples:
                    return
            page = self._store.fetch_games_before(page[-1].ingest_seq, _PAGE_SIZE)

    def _run_rounds(self, config: TrainingConfig, model_path: Path) -> dict[str, int | float]:
        schedule = config.schedule
        overrides = training_overrides(config)
        pending_games = total_committed = 0
        total_results: Counter[int] = Counter()
        for round_number in range(1, schedule.max_rounds + 1):
            self._round = round_number
            self._source_run_id = None
            self._games_committed = self._last_committed_seq = 0
            started = monotonic()
            self._phase = "selfplay"
            result = collect_live(
                self._store,
                games=schedule.games_per_round,
                engine_path=config.selfplay.engine_path,
                config_path=config.selfplay.config_path,
                model_path=model_path,
                overrides=overrides,
                on_source_registered=partial(self._source_registered, model_path=model_path),
                on_game_committed=self._game_committed,
            )
            self._event(
                "selfplay_finished",
                games_committed=result.games_committed,
                last_ingest_seq=result.last_ingest_seq,
            )

            self._phase = "refresh_pool"
            results, games_with_samples = self._refresh_pool(
                result.last_ingest_seq, result.source.source_run_id
            )
            pending_games += games_with_samples
            status = self._learner.pool_status()
            round_start_step = self._step
            new_games_ready = pending_games >= schedule.min_new_games_before_train
            trained = status.ready and new_games_ready
            if trained:
                self._train(schedule.train_steps_per_round)
                pending_games = 0
            else:
                self._event(
                    "training_skipped",
                    new_games=pending_games,
                    min_new_games_before_train=schedule.min_new_games_before_train,
                    sample_count=status.sample_count,
                    min_train_samples=config.learner.min_train_samples,
                    pool_ready=status.ready,
                    new_games_ready=new_games_ready,
                )

            last_round = round_number == schedule.max_rounds
            checkpoint_due = round_number % schedule.checkpoint_every_rounds == 0
            if self._step > self._last_checkpoint_step and (checkpoint_due or last_round):
                self._save_checkpoint("periodic" if checkpoint_due else "final")
            if trained and not last_round:
                self._export_model(model_path)

            self._phase = "log_round"
            stats = _result_metrics(result.games_committed, results)
            train_steps = self._step - round_start_step
            duration = monotonic() - started
            self._log.log_round(round_number, {
                **{f"selfplay/{name}": value for name, value in stats.items()},
                "selfplay/games_with_samples": games_with_samples,
                "replay/sample_count": status.sample_count,
                "replay/game_count": status.game_count,
                "round/train_steps": train_steps,
                "round/duration_seconds": duration,
            })
            self._log.flush()
            self._event(
                "round_finished",
                **stats,
                games_with_samples=games_with_samples,
                sample_count=status.sample_count,
                game_count=status.game_count,
                train_steps=train_steps,
                duration_seconds=duration,
            )
            total_committed += result.games_committed
            total_results.update(results)

        return {"rounds": schedule.max_rounds, **_result_metrics(total_committed, total_results)}

    def _refresh_pool(self, end_seq: int, source_run_id: UUID) -> tuple[Counter[int], int]:
        results: Counter[int] = Counter()
        games_with_samples = 0
        while self._cursor < end_seq:
            page = self._store.fetch_games_after(self._cursor, _PAGE_SIZE)
            if not page or page[0].ingest_seq > end_seq:
                raise RuntimeError(f"Cannot read through ingest_seq={end_seq} after {self._cursor}.")
            for game in page:
                if game.ingest_seq > end_seq:
                    break
                replayed = self._replay_game(game, position="newest")
                self._cursor = game.ingest_seq
                if replayed is not None and game.source.source_run_id == source_run_id:
                    black_result, has_samples = replayed
                    results[black_result] += 1
                    games_with_samples += has_samples
        return results, games_with_samples

    def _replay_game(
        self, game: StoredGame, *, position: Literal["oldest", "newest"]
    ) -> tuple[int, bool] | None:
        try:
            record = parse_record(game.record_text)
            rules = self._config.go
            _check_game_rules(record, game.source, rules)
            if record.komi != rules.komi:
                record = replace(record, komi=rules.komi)
            samples = replay_record(record, ko_rule=rules.ko_rule)
        except (RecordParseError, ReplayError) as error:
            self._event(
                "game_skipped", game_id=game.game_id, ingest_seq=game.ingest_seq,
                source_run_id=game.source.source_run_id, reason=str(error),
            )
            return None
        has_samples = samples.features.shape[0] > 0
        if has_samples:
            self._learner.add_samples(samples, position=position)
        return record.black_result, has_samples

    def _train(self, steps: int) -> None:
        self._phase = "warmup" if self._round is None else "train"
        self._event("training_started", steps=steps, phase=self._phase)
        for _ in range(steps):
            with _defer_sigint():
                self._checkpoint_safe = False
                metrics = self._learner.step()
                if metrics is None:
                    raise RuntimeError("The replay pool stopped being ready during training.")
                self._step = metrics.step
                self._checkpoint_safe = True
                self._log.log_step(metrics)

    def _save_checkpoint(self, reason: Literal["warmup", "periodic", "final", "interrupt"]) -> None:
        if reason != "interrupt":
            self._phase = "save_checkpoint"
        path = self._log.checkpoints / f"step-{self._step:06d}.pt"
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")
        with _defer_sigint(raise_after=reason != "interrupt"):
            self._learner.save_checkpoint(path)
            self._last_checkpoint_step = self._step
            self._last_checkpoint_path = path
            self._event("checkpoint_saved", path=path, reason=reason)

    def _export_model(self, path: Path) -> None:
        self._phase = "export_model"
        model = cast(Transformer, self._learner.get_inference_model())
        export_minizero_torchscript(model, path)
        self._event("model_exported", path=path)

    def _source_registered(self, source: SourceRun, *, model_path: Path) -> None:
        self._source_run_id = source.source_run_id
        self._event("selfplay_started", model_path=model_path, model_sha256=source.model_sha256)

    def _game_committed(self, _game_id: UUID, ingest_seq: int) -> None:
        self._games_committed += 1
        self._last_committed_seq = ingest_seq

    def _event(self, name: str, **fields: Any) -> None:
        context = {"round": self._round, "step": self._step, "source_run_id": self._source_run_id}
        self._log.event(name, **{**context, **fields})

    def _interrupt(self, error: KeyboardInterrupt) -> TrainingInterrupted:
        checkpoint_error = None
        if not self._checkpoint_safe:
            checkpoint_error = "Learner state is not ready to save."
        elif self._last_checkpoint_path is None or self._step > self._last_checkpoint_step:
            try:
                self._save_checkpoint("interrupt")
            except (Exception, KeyboardInterrupt) as save_error:
                checkpoint_error = str(save_error) or type(save_error).__name__
        self._abort(
            "run_interrupted", error, checkpoint_path=self._last_checkpoint_path,
            checkpoint_error=checkpoint_error,
        )
        return TrainingInterrupted(self._last_checkpoint_path, checkpoint_error)

    def _abort(self, name: str, error: BaseException, **fields: Any) -> None:
        with suppress(Exception):
            self._event(
                name, phase=self._phase, error=str(error) or type(error).__name__,
                ingest_seq=self._cursor, games_committed=self._games_committed,
                last_ingest_seq=self._last_committed_seq, **fields,
            )
        with suppress(Exception):
            self._log.close()


def _latest_checkpoint(directory: Path) -> Path | None:
    candidates = []
    for path in directory.glob("step-*.pt"):
        match = re.fullmatch(r"step-([0-9]+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match[1]), path))
    return max(candidates)[1] if candidates else None


@contextmanager
def _defer_sigint(*, raise_after: bool = True) -> Iterator[None]:
    """Finish the update or checkpoint write before honoring the CLI's Ctrl+C."""
    if (
        current_thread() is not main_thread()
        or signal.getsignal(signal.SIGINT) is not signal.default_int_handler
    ):
        yield
        return

    interrupted = False

    def remember_interrupt(_signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, remember_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    if interrupted and raise_after:
        raise KeyboardInterrupt


def _check_game_rules(record: GameRecord, source: SourceRun, go: GoRulesConfig) -> None:
    settings = parse_effective_config(source.effective_config)
    try:
        board_size = int(settings["env_board_size"])
        ko_rule = settings["env_go_ko_rule"]
        komi = float(settings["env_go_komi"])
    except (KeyError, ValueError) as error:
        raise ReplayError(f"Invalid Go rules in the source configuration: {error}.") from error

    if (
        board_size != go.board_size
        or ko_rule != go.ko_rule
        or komi not in (go.komi, effective_float(go.komi))
    ):
        raise ReplayError(
            f"Source rules ({board_size}, {komi}, {ko_rule}) do not match "
            f"the training rules ({go.board_size}, {go.komi}, {go.ko_rule})."
        )
    # Go writes the komi into the record with six decimal places.
    record_komi = float(f"{c_float(go.komi).value:.6f}")
    if record.board_size != go.board_size or record.komi not in (go.komi, record_komi):
        raise ReplayError(
            f"Record board size and komi ({record.board_size}, {record.komi}) "
            f"do not match the training rules ({go.board_size}, {go.komi})."
        )


def _result_metrics(committed: int, results: Counter[int]) -> dict[str, int | float]:
    scored = results.total()
    stats: dict[str, int | float] = {"games_committed": committed, "games_scored": scored}
    for result, count_tag, rate_tag in (
        (1, "black_wins", "black_win_rate"),
        (-1, "white_wins", "white_win_rate"),
        (0, "draws", "draw_rate"),
    ):
        count = results[result]
        stats[count_tag] = count
        if scored:
            stats[rate_tag] = count / scored
    return stats
