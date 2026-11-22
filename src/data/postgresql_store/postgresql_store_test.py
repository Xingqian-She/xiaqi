import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from uuid import uuid4

from .postgresql_store import IngestConflictError, PostgreSQLStore, SourceRun


DATABASE_URL = os.environ.get("XIAQI_TEST_DATABASE_URL")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
RECORD = "SelfPlay true 2 2 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1];B[81]P[81:1];W[81]P[81:1]) #"


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
        self.source = SourceRun(uuid4(), "test-pool", "a" * 64, "b" * 40, "env_board_size=9\n")
        self.store.register_source(self.source)

    def _connect(self):
        import psycopg

        return psycopg.connect(DATABASE_URL, autocommit=True, options=f"-c search_path={self.schema}")

    def test_retry_returns_committed_sequence_and_preserves_record(self) -> None:
        game_id = uuid4()
        self.assertEqual(self.store.append_game(self.source.pool_id, game_id, self.source.source_run_id, RECORD), 1)

        with self._connect() as reader:
            stored = reader.execute(
                "SELECT record_text, ingest_seq FROM games WHERE game_id = %s", (game_id,)
            ).fetchone()
        self.assertEqual(stored, (RECORD, 1))

        self.store.register_source(self.source)
        self.assertEqual(self.store.append_game(self.source.pool_id, game_id, self.source.source_run_id, RECORD), 1)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM games").fetchone(), (1,))

    def test_conflicts_preserve_source_and_pool_counters(self) -> None:
        from psycopg.errors import ForeignKeyViolation

        game_id = uuid4()
        self.store.append_game(self.source.pool_id, game_id, self.source.source_run_id, RECORD)
        other = replace(self.source, source_run_id=uuid4(), pool_id="other-pool")
        self.store.register_source(other)

        with self.assertRaises(IngestConflictError):
            self.store.register_source(replace(self.source, model_sha256="c" * 64))
        with self.assertRaises(IngestConflictError):
            self.store.append_game(self.source.pool_id, game_id, self.source.source_run_id, RECORD + " ")
        with self.assertRaises(IngestConflictError):
            self.store.append_game(other.pool_id, game_id, other.source_run_id, RECORD)
        with self.assertRaises(ForeignKeyViolation):
            self.store.append_game(other.pool_id, uuid4(), self.source.source_run_id, RECORD)

        self.store.register_source(self.source)
        self.assertEqual(self.store.append_game(self.source.pool_id, uuid4(), self.source.source_run_id, RECORD), 2)
        self.assertEqual(self.store.append_game(other.pool_id, uuid4(), other.source_run_id, RECORD), 1)

    def test_concurrent_writers_share_one_pool_sequence(self) -> None:
        duplicate, second = uuid4(), uuid4()
        ready = Barrier(3, timeout=10)

        def append(game_id):
            with self._connect() as connection:
                store = PostgreSQLStore(connection)
                ready.wait()
                return store.append_game(self.source.pool_id, game_id, self.source.source_run_id, RECORD)

        with ThreadPoolExecutor(max_workers=3) as writers:
            first, retried, distinct = writers.map(append, (duplicate, duplicate, second))
        self.assertEqual(first, retried)
        self.assertEqual({first, distinct}, {1, 2})
        self.assertEqual(self.connection.execute("SELECT last_ingest_seq FROM pools").fetchone(), (2,))
