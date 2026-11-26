from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class SourceRun:
    source_run_id: UUID
    model_sha256: str
    minizero_commit: str
    effective_config: str
