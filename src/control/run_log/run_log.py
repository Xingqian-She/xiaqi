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
    def __init__(self, runs_dir: Path, on_event: EventCallback | None = None) -> None:
        self.run_id = uuid4()
        self.directory = runs_dir / str(self.run_id)
        self.checkpoints = self.directory / "checkpoints"
        self.selfplay_models = self.directory / "selfplay_models"
        self._started_at = _timestamp()
        self._on_event = on_event
        self._events: TextIO | None = None
        self._writer: SummaryWriter | None = None

    def open(self) -> None:
        self.directory.mkdir(parents=True)
        self._events = (self.directory / "events.jsonl").open("x", encoding="utf-8", buffering=1)
        self.checkpoints.mkdir()
        self.selfplay_models.mkdir()

        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(self.directory / "tensorboard"))

    def write_metadata(self, **fields: Any) -> None:
        data = {"run_id": str(self.run_id), "started_at": self._started_at, **fields}
        (self.directory / "run.json").write_text(_json(data, indent=2) + "\n", encoding="utf-8")

    def event(self, name: str, **fields: Any) -> None:
        if self._events is None:
            raise RuntimeError("Run events are closed.")
        event = {"event": name, "run_id": str(self.run_id), "timestamp": _timestamp(), **fields}
        self._events.write(_json(event) + "\n")
        if self._on_event is not None:
            self._on_event(event)

    def log_step(self, metrics: StepMetrics) -> None:
        self.add_scalars(metrics.step, {
            "train/policy_loss": metrics.policy_loss,
            "train/value_loss": metrics.value_loss,
            "train/total_loss": metrics.total_loss,
            "train/batch_size": metrics.batch_size,
            "train/replay_samples": metrics.replay_samples,
        })

    def add_scalars(self, step: int, scalars: Mapping[str, int | float]) -> None:
        if self._writer is None:
            raise RuntimeError("TensorBoard metrics are closed.")
        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close_metrics(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()

    def close(self) -> None:
        try:
            self.close_metrics()
        finally:
            if self._events is not None:
                self._events.close()
                self._events = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: dict[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(data, default=str, ensure_ascii=False, allow_nan=False, indent=indent)
