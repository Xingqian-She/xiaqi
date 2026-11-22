import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame
from ..postgresql_store.postgresql_store import PostgreSQLStore
from .postgresql_reader import PostgreSQLReader


DATABASE_URL = os.environ.get("XIAQI_TEST_DATABASE_URL")
SCHEMA_PATH = Path(__file__).parents[1] / "postgresql_store" / "schema.sql"
RECORD = "SelfPlay true 2 2 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1];B[81]P[81:1];W[81]P[81:1]) #"


class ReaderInputTest(unittest.TestCase):
    def test_requires_autocommit(self) -> None:
        with self.assertRaisesRegex(ValueError, "autocommit"):
            PostgreSQLReader(Mock(autocommit=False))

    def test_rejects_invalid_page_arguments_before_querying(self) -> None:
        connection = Mock(autocommit=True)
        reader = PostgreSQLReader(connection)
        for limit in (0, -1, True, 1.5, None):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "limit"):
                    reader.fetch_games_before("test-pool", None, limit)
                with self.assertRaisesRegex(ValueError, "limit"):
                    reader.fetch_games_after("test-pool", 0, limit)
        with self.assertRaisesRegex(ValueError, "before_seq"):
            reader.fetch_games_before("test-pool", -1, 1)
        with self.assertRaisesRegex(ValueError, "after_seq"):
            reader.fetch_games_after("test-pool", None, 1)
        connection.cursor.assert_not_called()


@unittest.skipUnless(DATABASE_URL, "Set XIAQI_TEST_DATABASE_URL to run PostgreSQL tests.")
class PostgreSQLReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        from psycopg import sql

        self.schema = f"reader_test_{uuid4().hex}"
        writer = self.enterContext(self._connect())
        schema = sql.Identifier(self.schema)
        writer.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        self.addCleanup(writer.execute, sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
        writer.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.store = PostgreSQLStore(writer)
        self.reader = PostgreSQLReader(self.enterContext(self._connect()))
        self.source = SourceRun(uuid4(), "reader's-pool", "a" * 64, "b" * 40, "env_board_size=9\n")
        self.store.register_source(self.source)

    def _connect(self):
        import psycopg

        return psycopg.connect(DATABASE_URL, autocommit=True, options=f"-c search_path={self.schema}")

    def _append_game(self, source: SourceRun, record_text: str = RECORD) -> StoredGame:
        game_id = uuid4()
        ingest_seq = self.store.append_game(source.pool_id, game_id, source.source_run_id, record_text)
        return StoredGame(game_id, ingest_seq, record_text, source)

    def test_fetch_before_joins_sources_and_pages_newest_first(self) -> None:
        second_source = replace(
            self.source, source_run_id=uuid4(), model_sha256="c" * 64, effective_config="env_board_size=9\nenv_komi=7\n"
        )
        self.store.register_source(second_source)
        first = self._append_game(self.source)
        second = self._append_game(self.source)
        third = self._append_game(second_source, RECORD + "\n")

        pool_id = self.source.pool_id
        newest = self.reader.fetch_games_before(pool_id, None, 2)
        self.assertEqual(newest, [third, second])
        older = self.reader.fetch_games_before(pool_id, newest[-1].ingest_seq, 2)
        self.assertEqual(older, [first])
        self.assertEqual(self.reader.fetch_games_before(pool_id, first.ingest_seq, 2), [])

    def test_fetch_after_pages_oldest_first_and_keeps_pools_separate(self) -> None:
        other = replace(self.source, source_run_id=uuid4(), pool_id="other-pool")
        self.store.register_source(other)
        other_game = self._append_game(other)
        first, second, third = [self._append_game(self.source) for _ in range(3)]

        pool_id = self.source.pool_id
        page = self.reader.fetch_games_after(pool_id, 0, 2)
        self.assertEqual(page, [first, second])
        self.assertEqual(self.reader.fetch_games_after(pool_id, page[-1].ingest_seq, 2), [third])
        self.assertEqual(self.reader.fetch_games_after(pool_id, third.ingest_seq, 2), [])
        self.assertEqual(self.reader.fetch_games_before(other.pool_id, None, 2), [other_game])

    def test_new_commit_during_backfill_is_available_to_incremental_read(self) -> None:
        first, second, third = [self._append_game(self.source) for _ in range(3)]
        pool_id = self.source.pool_id
        newest = self.reader.fetch_games_before(pool_id, None, 2)
        self.assertEqual(newest, [third, second])

        new_game = self._append_game(self.source)
        older = self.reader.fetch_games_before(pool_id, newest[-1].ingest_seq, 2)
        self.assertEqual(older, [first])
        self.assertEqual(self.reader.fetch_games_after(pool_id, newest[0].ingest_seq, 2), [new_game])

    def test_empty_and_unknown_pools_return_empty_pages(self) -> None:
        for pool_id in (self.source.pool_id, "unknown-pool"):
            with self.subTest(pool_id=pool_id):
                self.assertEqual(self.reader.fetch_games_before(pool_id, None, 2), [])
                self.assertEqual(self.reader.fetch_games_after(pool_id, 0, 2), [])
