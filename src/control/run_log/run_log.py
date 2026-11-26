"""保存训练运行的元数据、事件和 TensorBoard 指标。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO
from uuid import uuid4

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from ...data_types.step_metrics.step_metrics import StepMetrics

EventCallback = Callable[[dict[str, Any]], None]


class RunLog:
    def __init__(self, directory: Path, on_event: EventCallback | None = None) -> None:
        self.run_id = uuid4()
        self.directory = directory
        self.checkpoints = self.directory / "checkpoints"
        self.selfplay_models = self.directory / "selfplay_models"
        self._on_event = on_event
        self._events: TextIO | None = None
        self._train_writer: SummaryWriter | None = None
        self._round_writer: SummaryWriter | None = None
        self._round_offset = 0

    def open(self) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        path = self.directory / "events.jsonl"
        with path.open("a+b") as history:
            history.seek(0)
            complete_size = 0
            for line in history:
                if not line.endswith(b"\n"):
                    # A hard stop can leave a partial final event.
                    history.truncate(complete_size)
                    break
                self._round_offset += json.loads(line)["event"] == "round_finished"
                complete_size += len(line)
        self._events = path.open("a", encoding="utf-8", buffering=1)

    def open_metrics(self, train_step: int) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = self.directory / "tensorboard"
        self._train_writer = SummaryWriter(log_dir=str(log_dir / "train"), purge_step=train_step + 1)
        self._round_writer = SummaryWriter(log_dir=str(log_dir / "rounds"), purge_step=self._round_offset + 1)

    def event(self, name: str, **fields: Any) -> None:
        if self._events is None:
            raise RuntimeError("Run events are closed.")
        event = {"event": name, "run_id": str(self.run_id), "timestamp": _timestamp(), **fields}
        self._events.write(json.dumps(event, default=str, ensure_ascii=False, allow_nan=False) + "\n")
        if self._on_event is not None:
            self._on_event(event)

    def log_step(self, metrics: StepMetrics) -> None:
        self._add_scalars(self._train_writer, metrics.step, {
            "train/policy_loss": metrics.policy_loss,
            "train/value_loss": metrics.value_loss,
            "train/total_loss": metrics.total_loss,
            "train/batch_size": metrics.batch_size,
            "train/replay_samples": metrics.replay_samples,
        })

    def log_round(self, number: int, scalars: Mapping[str, int | float]) -> None:
        self._add_scalars(self._round_writer, self._round_offset + number, scalars)

    @staticmethod
    def _add_scalars(
        writer: SummaryWriter | None, step: int, scalars: Mapping[str, int | float]
    ) -> None:
        if writer is None:
            raise RuntimeError("TensorBoard metrics are closed.")
        for tag, value in scalars.items():
            writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        for writer in (self._train_writer, self._round_writer):
            if writer is not None:
                writer.flush()

    def close_metrics(self) -> None:
        train_writer, round_writer = self._train_writer, self._round_writer
        self._train_writer = self._round_writer = None
        try:
            if train_writer is not None:
                train_writer.close()
        finally:
            if round_writer is not None:
                round_writer.close()

    def close(self) -> None:
        try:
            self.close_metrics()
        finally:
            if self._events is not None:
                self._events.close()
                self._events = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
