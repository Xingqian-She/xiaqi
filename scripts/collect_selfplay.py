"""Write existing or newly collected MiniZero game records to PostgreSQL."""

import argparse
import os
import sys
from pathlib import Path
from uuid import UUID

from src.control.selfplay_collector.selfplay_collector import (
    collect_live,
    import_records,
)
from src.data.postgresql.postgresql import PostgreSQLStore
from src.data_types.source_run.source_run import SourceRun


def _print_source(source: SourceRun) -> None:
    print(f"source_run_id={source.source_run_id}", flush=True)


def _print_game(game_id: UUID, ingest_seq: int) -> None:
    print(f"game_id={game_id} ingest_seq={ingest_seq}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record-file", type=Path, help="Read an existing MiniZero output file.")
    mode.add_argument("--games", type=int, help="Generate and store this many games.")

    live = parser.add_argument_group("live self-play")
    live.add_argument("--engine", type=Path, default=Path("build/minizero/go/minizero_go"))
    experiment = Path(os.environ.get("XIAQI_EXPERIMENT_DIR", "runs/go-9x9-az"))
    live.add_argument("--config", type=Path, default=experiment / "minizero_base.cfg")
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
        with psycopg.connect(database_url, autocommit=True) as connection:
            store = PostgreSQLStore(connection)
            if args.record_file is not None:
                source = SourceRun(
                    source_run_id=args.source_run_id,
                    model_sha256=args.model_sha256.lower(),
                    minizero_commit=args.minizero_commit.lower(),
                    effective_config=args.effective_config.read_text(encoding="utf-8"),
                )
                result = import_records(
                    store, source, args.record_file,
                    on_source_registered=_print_source, on_game_committed=_print_game,
                )
            else:
                result = collect_live(
                    store,
                    games=args.games,
                    engine_path=args.engine,
                    config_path=args.config,
                    model_path=args.model,
                    overrides=args.conf_str,
                    on_source_registered=_print_source,
                    on_game_committed=_print_game,
                )

        print(f"Processed {result.games_committed} complete games.")
        return 0
    except (OSError, RuntimeError, ValueError, psycopg.Error) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
