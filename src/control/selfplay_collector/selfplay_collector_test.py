import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4, uuid5

import pytest

from ...core.model.transformer.transformer import Transformer
from ...data_types.source_run.source_run import SourceRun
from ...integrations.minizero.torchscript_export import export_minizero_torchscript
from ..config_io.config_io import load_training_config
from . import selfplay_collector as collector

_ROOT = Path(__file__).resolve().parents[3]
_RECORD = "SelfPlay true 0 0 -1 (;GM[go_9x9]SZ[9]KM[7]RE[-1]DLEN[0--1]) #"


class _Store:
    def __init__(self):
        self.sources = {}
        self.games = {}

    def register_source(self, source):
        assert self.sources.setdefault(source.source_run_id, source) == source

    def append_game(self, pool_id, game_id, source_run_id, record):
        assert self.sources[source_run_id].pool_id == pool_id
        existing = self.games.setdefault(game_id, (record, len(self.games) + 1))
        assert existing[0] == record
        return existing[1]


def test_import_reuses_source_and_game_ids_after_retry(tmp_path):
    store = _Store()
    source = SourceRun(uuid4(), "test", "a" * 64, "b" * 40, "env_board_size=9")
    path = tmp_path / "selfplay.txt"
    path.write_text(f"ordinary output\n{_RECORD}\nSelfPlay false intermediate #\n{_RECORD}\n",
                    encoding="utf-8")
    callbacks = []
    for _ in range(2):
        result = collector.import_records(
            store, source, path, on_source_registered=callbacks.append,
            on_game_committed=lambda game_id, seq: callbacks.append((game_id, seq)),
        )
        assert (result.source, result.games_committed, result.last_ingest_seq) == (source, 2, 2)

    expected = [(uuid5(source.source_run_id, str(index)), index) for index in (1, 2)]
    assert callbacks == [source, *expected] * 2
    assert list(store.games) == [game_id for game_id, _ in expected]


def test_live_validates_torchscript_and_effective_config(tmp_path, monkeypatch):
    config = load_training_config(_ROOT / "configs/training/go_9x9_smoke.toml")
    config = replace(config, alphazero=replace(config.alphazero, action_selection="max_count"))
    overrides = collector.training_overrides(config)
    model = tmp_path / "weights.pt"
    export_minizero_torchscript(Transformer.from_config(config.model), model)
    (tmp_path / "engine.commit").write_text("a" * 40, encoding="ascii")
    monkeypatch.chdir(tmp_path)

    commands = []
    config_matches = True

    def render(command):
        commands.append(command)
        lines = command[-1].split(":")
        if not config_matches:
            lines = ["actor_num_simulation=1" if line == "actor_num_simulation=2" else line for line in lines]
        return "\n".join(f"{line} # effective value" for line in lines)

    @contextmanager
    def selfplay(command):
        assert command == commands[-1]
        assert store.sources
        yield iter(("noise", "SelfPlay false intermediate #", _RECORD, _RECORD))

    monkeypatch.setattr(collector, "read_effective_config", render)
    monkeypatch.setattr(collector, "selfplay_stdout", selfplay)
    store = _Store()
    arguments = dict(
        pool_id="test", games=1, engine_path=Path("engine"),
        config_path=config.selfplay.config_path, model_path=Path(model.name), overrides=overrides,
    )
    result = collector.collect_live(store, **arguments)
    assert result.games_committed == result.last_ingest_seq == 1
    assert result.source.model_sha256 == hashlib.sha256(model.read_bytes()).hexdigest()
    options = collector.parse_effective_config(result.source.effective_config)
    assert options["actor_select_action_by_count"] == "true"
    assert options["actor_select_action_by_softmax_count"] == "false"
    assert options["zero_actor_intermediate_sequence_length"] == "0"
    assert commands[-1][-1].endswith("nn_file_name=weights.pt")

    config_matches = False
    with pytest.raises(ValueError, match="MiniZero effective actor_num_simulation"):
        collector.collect_live(store, **arguments)
    with pytest.raises(ValueError, match="Incompatible MiniZero model"):
        collector.collect_live(store, **{**arguments, "overrides": {**overrides, "env_board_size": 13}})
    assert len(store.sources) == len(store.games) == 1
