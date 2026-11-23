from dataclasses import dataclass

from ..source_run.source_run import SourceRun


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    """一次采集或导入确认提交的棋谱及来源。"""

    games_committed: int
    last_ingest_seq: int
    source: SourceRun
