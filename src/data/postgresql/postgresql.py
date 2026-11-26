"""Read and write complete MiniZero games in an experiment's PostgreSQL database."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, LiteralString
from uuid import UUID

from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame

if TYPE_CHECKING:
    from psycopg import Connection


_SELECT_GAMES = """
    SELECT g.game_id, g.ingest_seq, g.record_text,
           s.source_run_id, s.model_sha256, s.minizero_commit, s.effective_config
    FROM games AS g
    JOIN source_runs AS s USING (source_run_id)
"""


class IngestConflictError(ValueError):
    """A source or game ID was reused with different data."""


class PostgreSQLStore:
    """Use one autocommit connection outside any caller-managed transaction."""

    def __init__(self, connection: Connection) -> None:
        if not connection.autocommit:
            raise ValueError("PostgreSQLStore requires an autocommit connection.")
        self._connection = connection

    def register_source(self, source: SourceRun) -> None:
        """Register immutable source metadata."""
        metadata = (source.model_sha256, source.minizero_commit, source.effective_config)
        with self._connection.transaction():
            self._connection.execute(
                """
                INSERT INTO source_runs (source_run_id, model_sha256, minizero_commit, effective_config)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_run_id) DO NOTHING
                """,
                (source.source_run_id, *metadata),
            )
            existing = self._connection.execute(
                """
                SELECT model_sha256, minizero_commit, effective_config
                FROM source_runs WHERE source_run_id = %s
                """,
                (source.source_run_id,),
            ).fetchone()
            if existing != metadata:
                raise IngestConflictError(f"Source {source.source_run_id} has different metadata.")

    def append_game(self, game_id: UUID, source_run_id: UUID, record_text: str) -> int:
        """Commit a game and return its ingest sequence, including on a retry."""
        with self._connection.transaction():
            # Hold the counter lock through commit so readers see games in sequence order.
            counter = self._connection.execute(
                "SELECT last_ingest_seq FROM database_state WHERE singleton FOR NO KEY UPDATE",
            ).fetchone()
            if counter is None:
                raise RuntimeError("The experiment database is missing its ingest counter.")
            next_seq = counter[0] + 1

            inserted = self._connection.execute(
                """
                INSERT INTO games (game_id, source_run_id, ingest_seq, record_text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (game_id) DO NOTHING
                RETURNING ingest_seq
                """,
                (game_id, source_run_id, next_seq, record_text),
            ).fetchone()
            if inserted is None:
                existing = self._connection.execute(
                    "SELECT source_run_id, record_text, ingest_seq FROM games WHERE game_id = %s",
                    (game_id,),
                ).fetchone()
                if existing is None or existing[:2] != (source_run_id, record_text):
                    raise IngestConflictError(f"Game {game_id} has a different source or record.")
                return existing[2]

            self._connection.execute(
                "UPDATE database_state SET last_ingest_seq = %s WHERE singleton",
                (next_seq,),
            )
            return next_seq

    def fetch_games_before(self, before_seq: int | None, limit: int) -> list[StoredGame]:
        """Read older games newest first. None starts at the latest game."""
        if before_seq is None:
            return self._fetch_page("ORDER BY g.ingest_seq DESC", (), limit)
        before_seq = _page_integer(before_seq, "before_seq", minimum=0)
        return self._fetch_page("WHERE g.ingest_seq < %s ORDER BY g.ingest_seq DESC", (before_seq,), limit)

    def fetch_games_after(self, after_seq: int, limit: int) -> list[StoredGame]:
        """Read games with a sequence greater than after_seq, oldest first."""
        after_seq = _page_integer(after_seq, "after_seq", minimum=0)
        return self._fetch_page("WHERE g.ingest_seq > %s ORDER BY g.ingest_seq ASC", (after_seq,), limit)

    def _fetch_page(self, clause: LiteralString, params: tuple[int, ...], limit: int) -> list[StoredGame]:
        limit = _page_integer(limit, "limit", minimum=1)
        with self._connection.cursor() as cursor:
            cursor.execute(_SELECT_GAMES + clause + " LIMIT %s", (*params, limit))
            rows = cursor.fetchall()
        return [
            StoredGame(game_id, ingest_seq, record_text, SourceRun(*source_fields))
            for game_id, ingest_seq, record_text, *source_fields in rows
        ]


def _page_integer(value: int, name: str, *, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)
