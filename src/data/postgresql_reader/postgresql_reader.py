"""Read pages of committed MiniZero records from one PostgreSQL pool."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, LiteralString

from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame

if TYPE_CHECKING:
    from psycopg import Connection


_SELECT_GAMES = """
    SELECT g.game_id, g.ingest_seq, g.record_text,
           s.source_run_id, s.pool_id, s.model_sha256, s.minizero_commit, s.effective_config
    FROM games AS g
    JOIN source_runs AS s USING (pool_id, source_run_id)
    WHERE g.pool_id = %s
"""


class PostgreSQLReader:
    """Use a caller-managed autocommit connection outside a transaction."""

    def __init__(self, connection: Connection) -> None:
        if not connection.autocommit:
            raise ValueError("PostgreSQLReader requires an autocommit connection.")
        self._connection = connection

    def fetch_games_before(self, pool_id: str, before_seq: int | None, limit: int) -> list[StoredGame]:
        """Read older games newest first. None starts at the latest game."""
        if before_seq is None:
            return self._fetch_page("ORDER BY g.ingest_seq DESC", (pool_id,), limit)
        before_seq = _page_integer(before_seq, "before_seq", minimum=0)
        return self._fetch_page("AND g.ingest_seq < %s ORDER BY g.ingest_seq DESC", (pool_id, before_seq), limit)

    def fetch_games_after(self, pool_id: str, after_seq: int, limit: int) -> list[StoredGame]:
        """Read games with a sequence greater than after_seq, oldest first."""
        after_seq = _page_integer(after_seq, "after_seq", minimum=0)
        return self._fetch_page("AND g.ingest_seq > %s ORDER BY g.ingest_seq ASC", (pool_id, after_seq), limit)

    def _fetch_page(self, clause: LiteralString, params: tuple[str] | tuple[str, int], limit: int) -> list[StoredGame]:
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
