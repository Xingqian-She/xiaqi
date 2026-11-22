"""Store complete MiniZero records and their source metadata in PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from psycopg import Connection


@dataclass(frozen=True)
class SourceRun:
    source_run_id: UUID
    pool_id: str
    model_sha256: str
    minizero_commit: str
    effective_config: str


class IngestConflictError(ValueError):
    """A source or game ID was reused with different data."""


class PostgreSQLStore:
    """Use one autocommit connection outside any caller-managed transaction."""

    def __init__(self, connection: Connection) -> None:
        if not connection.autocommit:
            raise ValueError("PostgreSQLStore requires an autocommit connection.")
        self._connection = connection

    def register_source(self, source: SourceRun) -> None:
        """Create the pool if needed and register immutable source metadata."""
        metadata = (source.pool_id, source.model_sha256, source.minizero_commit, source.effective_config)
        with self._connection.transaction():
            self._connection.execute(
                "INSERT INTO pools (pool_id) VALUES (%s) ON CONFLICT (pool_id) DO NOTHING",
                (source.pool_id,),
            )
            self._connection.execute(
                """
                INSERT INTO source_runs
                    (source_run_id, pool_id, model_sha256, minizero_commit, effective_config)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_run_id) DO NOTHING
                """,
                (source.source_run_id, *metadata),
            )
            existing = self._connection.execute(
                """
                SELECT pool_id, model_sha256, minizero_commit, effective_config
                FROM source_runs WHERE source_run_id = %s
                """,
                (source.source_run_id,),
            ).fetchone()
            if existing != metadata:
                raise IngestConflictError(f"Source {source.source_run_id} has different metadata.")

    def append_game(self, pool_id: str, game_id: UUID, source_run_id: UUID, record_text: str) -> int:
        """Commit one game and return its pool sequence, including on a retry."""
        with self._connection.transaction():
            # The pool row stays locked through commit, so later numbers cannot become visible first.
            counter = self._connection.execute(
                "SELECT last_ingest_seq FROM pools WHERE pool_id = %s FOR NO KEY UPDATE",
                (pool_id,),
            ).fetchone()
            if counter is None:
                raise ValueError(f"Unknown pool {pool_id!r}.")
            next_seq = counter[0] + 1

            inserted = self._connection.execute(
                """
                INSERT INTO games (game_id, pool_id, source_run_id, ingest_seq, record_text)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO NOTHING
                RETURNING ingest_seq
                """,
                (game_id, pool_id, source_run_id, next_seq, record_text),
            ).fetchone()
            if inserted is None:
                existing = self._connection.execute(
                    "SELECT pool_id, source_run_id, record_text, ingest_seq FROM games WHERE game_id = %s",
                    (game_id,),
                ).fetchone()
                if existing is None or existing[:3] != (pool_id, source_run_id, record_text):
                    raise IngestConflictError(f"Game {game_id} has a different pool, source or record.")
                return existing[3]

            self._connection.execute(
                "UPDATE pools SET last_ingest_seq = %s WHERE pool_id = %s",
                (next_seq, pool_id),
            )
            return next_seq
