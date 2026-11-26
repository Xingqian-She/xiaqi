"""Run serial MiniZero self-play and training in the dev container."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from src.control.training_controller.training_controller import (
    TrainingController,
    TrainingInterrupted,
)


def _print_progress(event: dict[str, Any]) -> None:
    name = event["event"]
    if name == "run_started":
        print(f"run_id={event['run_id']} run_dir={event['run_dir']}", flush=True)
    elif name == "selfplay_started":
        print(f"round={event['round']} source_run_id={event['source_run_id']}", flush=True)
    elif name == "training_started":
        print(f"{event['phase']} steps={event['steps']} starting_step={event['step']}", flush=True)
    elif name == "training_skipped":
        print(
            f"round={event['round']} training skipped, "
            f"new_games={event['new_games']}/{event['min_new_games_before_train']} "
            f"samples={event['sample_count']}/{event['min_train_samples']}",
            flush=True,
        )
    elif name == "round_finished":
        print(
            f"round={event['round']} games={event['games_committed']} "
            f"samples={event['sample_count']} train_steps={event['train_steps']} step={event['step']}",
            flush=True,
        )
    elif name == "run_finished":
        print(f"Finished {event['rounds']} rounds at step {event['step']}.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training TOML configuration.")
    parser.add_argument("--resume", action="store_true", help="Resume the experiment's latest checkpoint.")
    args = parser.parse_args(argv)

    database_url = os.environ.get("XIAQI_DATABASE_URL")
    if not database_url:
        parser.error("set XIAQI_DATABASE_URL to the PostgreSQL connection string")

    try:
        import psycopg
    except ImportError:
        parser.exit(1, "psycopg is required. Rebuild the dev image or install psycopg in your Python environment.\n")

    try:
        controller = TrainingController(args.config, on_event=_print_progress)
        with psycopg.connect(database_url, autocommit=True) as connection:
            controller.run(connection, resume=args.resume)
    except (OSError, RuntimeError, ValueError, psycopg.Error) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt as error:
        print("Training interrupted.", file=sys.stderr)
        if isinstance(error, TrainingInterrupted):
            if error.checkpoint_error is not None:
                print(f"Checkpoint save failed ({error.checkpoint_error})", file=sys.stderr)
            if error.checkpoint_path is not None:
                print(f"Saved checkpoint {error.checkpoint_path}. Resume the experiment with -Resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
