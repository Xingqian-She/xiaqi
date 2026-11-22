"""Write existing or newly collected MiniZero game records to PostgreSQL."""

import argparse
import hashlib
import os
import sys
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from src.data.minizero_selfplay.minizero_selfplay import (
    iter_complete_records,
    read_effective_config,
    selfplay_stdout,
)
from src.data.postgresql_store.postgresql_store import PostgreSQLStore
from src.data_types.source_run.source_run import SourceRun


def _append_records(store: PostgreSQLStore, source: SourceRun, records: Iterable[str]) -> int:
    count = 0
    for count, record in enumerate(records, start=1):
        game_id = uuid5(source.source_run_id, str(count))
        ingest_seq = store.append_game(source.pool_id, game_id, source.source_run_id, record)
        print(f"game_id={game_id} ingest_seq={ingest_seq}", flush=True)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-id", required=True, help="Destination experience pool.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record-file", type=Path, help="Read an existing MiniZero output file.")
    mode.add_argument("--games", type=int, help="Generate and store this many games.")

    live = parser.add_argument_group("live self-play")
    live.add_argument("--engine", type=Path, default=Path("build/minizero/go/minizero_go"))
    live.add_argument("--config", type=Path, default=Path("configs/minizero/go_9x9_az.cfg"))
    live.add_argument("--model", type=Path, default=Path("models/minizero/go_9x9_az_n200.pt"))
    live.add_argument("--conf-str", default="", help="MiniZero configuration overrides.")

    saved = parser.add_argument_group("source metadata required for --record-file")
    saved.add_argument("--source-run-id", type=UUID, help="Reuse this UUID to retry an import.")
    saved.add_argument("--model-sha256", help="SHA-256 of the model used to generate the record file.")
    saved.add_argument("--minizero-commit", help="Full commit hash of the MiniZero build.")
    saved.add_argument("--effective-config", type=Path, help="Path to the saved effective configuration.")
    args = parser.parse_args(argv)

    saved_metadata = (args.source_run_id, args.model_sha256, args.minizero_commit, args.effective_config)
    if args.record_file is not None:
        if any(value is None for value in saved_metadata):
            parser.error("--record-file requires --source-run-id, --model-sha256, --minizero-commit and --effective-config")
    elif args.games <= 0 or any(value is not None for value in saved_metadata):
        parser.error("--games requires a positive count and generates its own source metadata")

    database_url = os.environ.get("XIAQI_DATABASE_URL")
    if not database_url:
        parser.error("set XIAQI_DATABASE_URL to the PostgreSQL connection string")

    try:
        import psycopg
    except ImportError:
        parser.exit(1, "psycopg is required. Rebuild the dev image or install psycopg in your Python environment.\n")

    try:
        command = None
        if args.record_file is not None:
            source = SourceRun(
                source_run_id=args.source_run_id,
                pool_id=args.pool_id,
                model_sha256=args.model_sha256.lower(),
                minizero_commit=args.minizero_commit.lower(),
                effective_config=args.effective_config.read_text(encoding="utf-8"),
            )
        else:
            engine = args.engine.resolve()
            commit_file = engine.with_name(f"{engine.name}.commit")
            if not commit_file.is_file():
                raise ValueError(f"Missing {commit_file}. Build the engine with scripts/build_minizero.sh.")
            model_path = str(args.model)
            if model_path != model_path.strip() or any(char in model_path for char in ":#\r\n"):
                raise ValueError("The model path cannot contain ':', '#', line breaks or surrounding whitespace.")
            with args.model.open("rb") as model_file:
                model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
            overrides = ":".join(filter(None, (args.conf_str, f"nn_file_name={model_path}")))
            command = [str(engine), "-conf_file", str(args.config), "-conf_str", overrides]
            source = SourceRun(
                source_run_id=uuid4(),
                pool_id=args.pool_id,
                model_sha256=model_sha256,
                minizero_commit=commit_file.read_text(encoding="ascii").strip(),
                effective_config=read_effective_config(command),
            )

        with psycopg.connect(database_url, autocommit=True) as connection:
            store = PostgreSQLStore(connection)
            store.register_source(source)
            print(f"pool_id={source.pool_id} source_run_id={source.source_run_id}", flush=True)
            input_stream = selfplay_stdout(command) if command else args.record_file.open(encoding="utf-8")
            with input_stream as lines:
                count = _append_records(store, source, islice(iter_complete_records(lines), args.games))

        if args.games is not None and count < args.games:
            raise RuntimeError(f"MiniZero ended after {count} of {args.games} requested games.")
        if count == 0:
            raise ValueError(f"No complete SelfPlay records in {args.record_file}.")
        print(f"Processed {count} complete games.")
        return 0
    except (OSError, RuntimeError, ValueError, psycopg.Error) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
