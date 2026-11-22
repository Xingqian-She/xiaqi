"""Read complete-game messages from MiniZero's self-play output."""

import subprocess
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO


def iter_complete_records(lines: Iterable[str]) -> Iterator[str]:
    """Keep the complete SelfPlay messages, preserving their original text."""
    for line in lines:
        record = line.rstrip("\r\n")
        if record.startswith("SelfPlay true ") and record.rstrip().endswith(" #"):
            yield record


def read_effective_config(command: Sequence[str]) -> str:
    """Ask MiniZero to render the configuration after applying overrides."""
    with TemporaryDirectory(prefix="xiaqi-minizero-") as directory:
        config = Path(directory) / "effective.cfg"
        result = subprocess.run(
            [*command, "-gen", str(config)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            raise RuntimeError(f"MiniZero configuration failed:\n{result.stderr.strip()}")
        return config.read_text(encoding="utf-8")


@contextmanager
def selfplay_stdout(command: Sequence[str]) -> Generator[TextIO, None, None]:
    """Start MiniZero and drain its output during shutdown."""
    with subprocess.Popen(
        [*command, "-mode", "sp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ) as process:
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write("start\n")
            process.stdin.flush()
            yield process.stdout
        finally:
            if process.poll() is None:
                with suppress(BrokenPipeError):
                    process.stdin.write("quit\n")
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"MiniZero exited with status {process.returncode}.")
