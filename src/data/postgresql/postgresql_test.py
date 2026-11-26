import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock
from uuid import uuid4

from ...data_types.source_run.source_run import SourceRun
from ...data_types.stored_game.stored_game import StoredGame
from .postgresql import IngestConflictError, PostgreSQLStore


DATABASE_URL = os.environ.get("XIAQI_TEST_DATABASE_URL")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
RECORD = "SelfPlay true 2 2 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1];B[81]P[81:1];W[81]P[81:1]) #"


class StoreInputTest(unittest.TestCase):
    def test_requires_autocommit(self) -> None:
        with self.assertRaisesRegex(ValueError, "autocommit"):
            PostgreSQLStore(Mock(autocommit=False))

    def test_rejects_invalid_page_arguments_before_querying(self) -> None:
        connection = Mock(autocommit=True)
        store = PostgreSQLStore(connection)
        for limit in (0, -1, True, 1.5, None):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "limit"):
                    store.fetch_games_before(None, limit)
                with self.assertRaisesRegex(ValueError, "limit"):
                    store.fetch_games_after(0, limit)
        with self.assertRaisesRegex(ValueError, "before_seq"):
            store.fetch_games_before(-1, 1)
        with self.assertRaisesRegex(ValueError, "after_seq"):
            store.fetch_games_after(None, 1)
        connection.cursor.assert_not_called()


@unittest.skipUnless(DATABASE_URL, "Set XIAQI_TEST_DATABASE_URL to run PostgreSQL tests.")
class PostgreSQLStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        from psycopg import sql

        self.schema = f"ingest_test_{uuid4().hex}"
        self.connection = self.enterContext(self._connect())
        schema = sql.Identifier(self.schema)
        self.connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        self.addCleanup(self.connection.execute, sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
        self.connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.store = PostgreSQLStore(self.connection)
        self.reader = PostgreSQLStore(self.enterContext(self._connect()))
        self.source = SourceRun(uuid4(), "a" * 64, "b" * 40, "env_board_size=9\n")
        self.store.register_source(self.source)

    def _connect(self):
        import psycopg

        return psycopg.connect(DATABASE_URL, autocommit=True, options=f"-c search_path={self.schema}")

    def _append_game(self, source: SourceRun, record_text: str = RECORD) -> StoredGame:
        game_id = uuid4()
        ingest_seq = self.store.append_game(game_id, source.source_run_id, record_text)
        return StoredGame(game_id, ingest_seq, record_text, source)

    def test_retry_returns_committed_sequence_and_preserves_record(self) -> None:
        first = self._append_game(self.source)
        self.assertEqual(self.reader.fetch_games_before(None, 1), [first])
        self.store.register_source(self.source)
        self.assertEqual(self.store.append_game(first.game_id, self.source.source_run_id, RECORD), 1)
        self.assertEqual(self.reader.fetch_games_after(0, 2), [first])

    def test_conflicts_and_foreign_keys_preserve_the_counter(self) -> None:
        from psycopg.errors import ForeignKeyViolation

        first = self._append_game(self.source)
        other = replace(self.source, source_run_id=uuid4())
        self.store.register_source(other)
        with self.assertRaises(IngestConflictError):
            self.store.register_source(replace(self.source, model_sha256="c" * 64))
        with self.assertRaises(IngestConflictError):
            self.store.append_game(first.game_id, self.source.source_run_id, RECORD + " ")
        with self.assertRaises(IngestConflictError):
            self.store.append_game(first.game_id, other.source_run_id, RECORD)
        with self.assertRaises(ForeignKeyViolation):
            self.store.append_game(uuid4(), uuid4(), RECORD)
        self.assertEqual(self._append_game(other).ingest_seq, 2)
        self.assertEqual(self.reader.fetch_games_before(2, 1), [first])

    def test_concurrent_writers_share_the_database_sequence(self) -> None:
        duplicate, second = uuid4(), uuid4()
        ready = Barrier(3, timeout=10)

        def append(game_id):
            with self._connect() as connection:
                store = PostgreSQLStore(connection)
                ready.wait()
                return store.append_game(game_id, self.source.source_run_id, RECORD)

        with ThreadPoolExecutor(max_workers=3) as writers:
            first, retried, distinct = writers.map(append, (duplicate, duplicate, second))
        self.assertEqual(first, retried)
        self.assertEqual({first, distinct}, {1, 2})
        self.assertEqual(
            self.connection.execute("SELECT last_ingest_seq FROM database_state").fetchone(),
            (2,),
        )
        self.assertEqual([game.ingest_seq for game in self.reader.fetch_games_after(0, 3)], [1, 2])

    def test_fetch_before_joins_sources_and_pages_newest_first(self) -> None:
        second_source = replace(
            self.source, source_run_id=uuid4(), model_sha256="c" * 64,
            effective_config="env_board_size=9\nenv_go_komi=7\n",
        )
        self.store.register_source(second_source)
        first = self._append_game(self.source)
        second = self._append_game(self.source)
        third = self._append_game(second_source, RECORD + "\n")

        newest = self.reader.fetch_games_before(None, 2)
        self.assertEqual(newest, [third, second])
        self.assertEqual(self.reader.fetch_games_before(newest[-1].ingest_seq, 2), [first])
        self.assertEqual(self.reader.fetch_games_before(first.ingest_seq, 2), [])

    def test_fetch_after_pages_oldest_first(self) -> None:
        first, second, third = [self._append_game(self.source) for _ in range(3)]
        page = self.reader.fetch_games_after(0, 2)
        self.assertEqual(page, [first, second])
        self.assertEqual(self.reader.fetch_games_after(page[-1].ingest_seq, 2), [third])
        self.assertEqual(self.reader.fetch_games_after(third.ingest_seq, 2), [])

    def test_new_commit_during_backfill_is_available_to_incremental_read(self) -> None:
        first, second, third = [self._append_game(self.source) for _ in range(3)]
        newest = self.reader.fetch_games_before(None, 2)
        self.assertEqual(newest, [third, second])

        new_game = self._append_game(self.source)
        self.assertEqual(self.reader.fetch_games_before(newest[-1].ingest_seq, 2), [first])
        self.assertEqual(self.reader.fetch_games_after(newest[0].ingest_seq, 2), [new_game])

    def test_empty_database_returns_empty_pages(self) -> None:
        self.assertEqual(self.reader.fetch_games_before(None, 2), [])
        self.assertEqual(self.reader.fetch_games_after(0, 2), [])
