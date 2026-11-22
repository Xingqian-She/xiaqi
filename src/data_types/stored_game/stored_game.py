from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ..source_run.source_run import SourceRun


@dataclass(frozen=True)
class StoredGame:
    game_id: UUID
    ingest_seq: int
    record_text: str
    source: SourceRun
