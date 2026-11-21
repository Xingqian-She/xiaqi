"""Parse the native text records emitted by MiniZero Go."""

import math
import re
from collections.abc import Iterator

from .game_record.game_record import GameRecord
from .move.move import Move

_SELFPLAY_HEADER = re.compile(
    r"SelfPlay\s+(true|false)\s+([0-9]+)\s+([0-9]+)\s+(\S+)\s+"
)
_RECORD_START = re.compile(r"\(\s*;")
_PROPERTY = re.compile(r"\s*([A-Z]+)\s*\[((?:\\.|[^\\\]])*)\]", re.DOTALL)
_NODE_END = re.compile(r"\s*([;)])")
_ESCAPE = re.compile(r"\\(.)", re.DOTALL)


class RecordParseError(ValueError):
    """A record is malformed or outside the MiniZero Go format."""


def _read_nodes(text: str) -> Iterator[dict[str, str]]:
    start = _RECORD_START.match(text)
    if start is None:
        raise RecordParseError("Expected a MiniZero game starting with '(;'.")

    offset = start.end()
    while True:
        properties: dict[str, str] = {}
        while match := _PROPERTY.match(text, offset):
            name, value = match.groups()
            if name in properties:
                raise RecordParseError(
                    f"Duplicate {name} at character {match.start(1) + 1}."
                )
            if name in {"AB", "AW", "AE", "PL"}:
                raise RecordParseError(
                    f"Setup or turn override property {name} is unsupported."
                )
            properties[name] = _ESCAPE.sub(r"\1", value)
            offset = match.end()

        end = _NODE_END.match(text, offset)
        if end is None:
            raise RecordParseError(
                f"Expected a property or node delimiter at character {offset + 1}."
            )
        offset = end.end()
        if end[1] == ")":
            if offset != len(text):
                raise RecordParseError("Unexpected text after the game.")
            yield properties
            return
        yield properties


def _required(properties: dict[str, str], name: str) -> str:
    try:
        return properties[name]
    except KeyError as error:
        raise RecordParseError(f"Missing {name} property.") from error


def _read_index(value: str, field: str) -> int:
    value = value.strip()
    if not value.isascii() or not value.isdecimal():
        raise RecordParseError(f"{field} must be a nonnegative integer.")
    try:
        return int(value)
    except ValueError as error:
        raise RecordParseError(f"{field} is too large.") from error


def _read_number(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise RecordParseError(f"{field} must be a finite number.") from error
    if not math.isfinite(number):
        raise RecordParseError(f"{field} must be a finite number.")
    return number


def _read_action_id(value: str, pass_id: int, field: str) -> int:
    action_id = _read_index(value, field)
    if action_id > pass_id:
        raise RecordParseError(
            f"{field} is outside the action ID range 0 to {pass_id}."
        )
    return action_id


def _read_policy(value: str, pass_id: int) -> tuple[tuple[int, float], ...]:
    if not value.strip():
        return ()

    weights: dict[int, float] = {}
    for entry in value.split(","):
        raw_id, separator, raw_weight = entry.partition(":")
        if not separator:
            raise RecordParseError("P requires comma-separated action_id:weight pairs.")
        action_id = _read_action_id(raw_id, pass_id, "P action ID")
        if action_id in weights:
            raise RecordParseError(f"P contains duplicate action ID {action_id}.")
        weight = _read_number(raw_weight, "P weight")
        if weight < 0:
            raise RecordParseError("P weights must be nonnegative.")
        weights[action_id] = weight

    if not 0 < sum(weights.values()) < math.inf:
        raise RecordParseError("P weights must have a positive finite sum.")
    return tuple(weights.items())


def _read_move(properties: dict[str, str], pass_id: int, move_number: int) -> Move:
    try:
        move_keys = properties.keys() & {"B", "W"}
        if len(move_keys) != 1:
            raise RecordParseError("Expected exactly one B or W property.")
        (color,) = move_keys
        return Move(
            player=1 if color == "B" else 2,
            action_id=_read_action_id(properties[color], pass_id, color),
            policy_weights=_read_policy(properties.get("P", ""), pass_id),
            search_value=_read_number(properties["V"], "V")
            if "V" in properties
            else None,
            reward=_read_number(properties["R"], "R") if "R" in properties else None,
        )
    except RecordParseError as error:
        raise RecordParseError(f"Move {move_number} - {error}") from error


def _read_sample_range(value: str, move_count: int) -> range:
    value = value.strip()
    if not value or (value == "0--1" and move_count == 0):
        return range(move_count)
    first, _, last = value.partition("-")
    start = _read_index(first, "DLEN start")
    stop = _read_index(last, "DLEN end") + 1
    if not start < stop <= move_count:
        raise RecordParseError("DLEN must select a range within the recorded moves.")
    return range(start, stop)


def parse_record(text: str) -> GameRecord:
    """Parse one MiniZero Go game body or one full SelfPlay message."""
    text = text.strip()
    has_end_marker = text.endswith("#")
    if has_end_marker:
        text = text[:-1].rstrip()
    is_complete = True if has_end_marker else None

    header = None
    if text.startswith("SelfPlay"):
        header = _SELFPLAY_HEADER.match(text)
        if header is None or not has_end_marker:
            raise RecordParseError(
                "Expected 'SelfPlay true|false DATA_LEN GAME_LEN RESULT (...) #'."
            )
        is_complete = header[1] == "true"
        text = text[header.end() :]

    nodes = _read_nodes(text)
    root = next(nodes)
    if "B" in root or "W" in root:
        raise RecordParseError("Moves must follow the root metadata node.")
    board_size = _read_index(_required(root, "SZ"), "SZ")
    if not 1 <= board_size <= 19:
        raise RecordParseError("SZ must be between 1 and 19 for MiniZero Go.")
    game_name = f"go_{board_size}x{board_size}"
    if _required(root, "GM") != game_name:
        raise RecordParseError(f"Expected GM[{game_name}] for SZ[{board_size}].")
    komi = _read_number(_required(root, "KM"), "KM")
    black_result = _read_number(_required(root, "RE"), "RE")
    if black_result not in (-1, 0, 1):
        raise RecordParseError("Go RE must be -1, 0 or +1 from Black's perspective.")

    pass_id = board_size**2
    moves = tuple(
        _read_move(node, pass_id, number) for number, node in enumerate(nodes, 1)
    )
    sample_range = _read_sample_range(root.get("DLEN", ""), len(moves))

    if header is not None:
        _, data_length, game_length, result = header.groups()
        if _read_index(game_length, "SelfPlay game length") != len(moves):
            raise RecordParseError(
                "SelfPlay game length disagrees with the move count."
            )
        if _read_index(data_length, "SelfPlay data length") != len(sample_range):
            raise RecordParseError("SelfPlay data length disagrees with DLEN.")
        if _read_number(result, "SelfPlay result") != black_result:
            raise RecordParseError("SelfPlay result disagrees with RE.")

    return GameRecord(
        board_size=board_size,
        komi=komi,
        black_result=int(black_result),
        moves=moves,
        sample_range=sample_range,
        model_name=root.get("EV"),
        is_complete=is_complete,
    )
