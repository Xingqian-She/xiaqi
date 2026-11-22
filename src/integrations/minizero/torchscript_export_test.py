import hashlib
import os
import signal
import subprocess
import sys
import unittest
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import torch

from ...core.model.transformer.transformer import Transformer
from ...data.minizero_record.minizero_record import parse_record
from ...data_types.transformer_config.transformer_config import TransformerConfig
from .torchscript_export import export_minizero_torchscript


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY_CONFIG = TransformerConfig(
    board_size=9,
    in_channels=18,
    model_dim=8,
    num_heads=2,
    ffn_hidden_dim=8,
    num_layers=1,
    value_hidden_dim=4,
    num_stem_blocks=1,
)


def _tiny_model() -> Transformer:
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        return Transformer.from_config(_TINY_CONFIG)


class TorchScriptExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _tiny_model()
        self.path = Path(self.enterContext(TemporaryDirectory())) / "tiny_go_9x9.pt"

    @torch.inference_mode()
    def test_round_trip_matches_eager_and_exposes_minizero_metadata(self) -> None:
        reference = deepcopy(self.model).eval()
        export_minizero_torchscript(self.model, self.path)
        loaded = torch.jit.load(self.path, map_location="cpu")

        metadata = {
            "get_game_name": "go_9x9",
            "get_type_name": "alphazero",
            "get_num_input_channels": 18,
            "get_input_channel_height": 9,
            "get_input_channel_width": 9,
            "get_num_hidden_channels": 8,
            "get_hidden_channel_height": 9,
            "get_hidden_channel_width": 9,
            "get_num_blocks": 1,
            "get_action_size": 82,
            "get_num_value_hidden_channels": 4,
            "get_discrete_value_size": 1,
        }
        for method, expected in metadata.items():
            with self.subTest(method=method):
                actual = getattr(loaded, method)()
                self.assertIs(type(actual), type(expected))
                self.assertEqual(actual, expected)

        board = torch.zeros(2, 18, 9, 9)
        board[0, 16].fill_(1)
        board[1, 17].fill_(1)
        board[1, 1, 4, 4] = 1
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                batch = board[:batch_size]
                logits, value = reference(batch)
                outputs = loaded(batch)
                self.assertEqual(set(outputs), {"policy_logit", "policy", "value"})
                for tensor in outputs.values():
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertTrue(tensor.is_contiguous())

                self.assertTrue(value.ne(0).all().item())
                value[1:] *= -1
                torch.testing.assert_close(outputs["value"], value)
                torch.testing.assert_close(outputs["policy_logit"], logits)
                torch.testing.assert_close(outputs["policy"], logits.softmax(dim=1))
                torch.testing.assert_close(outputs["policy"].sum(dim=1), torch.ones(batch_size))

    def test_exports_a_float32_eval_copy_and_preserves_training_state(self) -> None:
        self.model.double()
        self.model.encoder.eval()
        gradient = torch.ones_like(self.model.cls_token)
        self.model.cls_token.grad = gradient.clone()
        original_weights = deepcopy(self.model.state_dict())
        original_modes = [module.training for module in self.model.modules()]

        export_minizero_torchscript(self.model, self.path)

        torch.testing.assert_close(self.model.state_dict(), original_weights, rtol=0, atol=0)
        torch.testing.assert_close(self.model.cls_token.grad, gradient)
        self.assertEqual([module.training for module in self.model.modules()], original_modes)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

        loaded = torch.jit.load(self.path)
        self.assertFalse(loaded.training)
        for parameter in loaded.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.device.type, "cpu")
            self.assertFalse(parameter.requires_grad)

    def test_rejects_models_outside_minizero_go_input_limits(self) -> None:
        for config, message in (
            (replace(_TINY_CONFIG, in_channels=3), "requires 18 input channels"),
            (replace(_TINY_CONFIG, board_size=21), "cannot exceed 19"),
        ):
            with self.subTest(config=config):
                model = Transformer.from_config(config)
                with self.assertRaisesRegex(ValueError, message):
                    export_minizero_torchscript(model, self.path)
        self.assertFalse(self.path.exists())

    def test_failed_save_preserves_the_previous_model(self) -> None:
        export_minizero_torchscript(self.model, self.path)
        previous = self.path.read_bytes()

        def fail_save(_model, file) -> None:
            file.write(b"incomplete model")
            raise RuntimeError("simulated save failure")

        with patch("torch.jit.save", side_effect=fail_save):
            with self.assertRaisesRegex(RuntimeError, "simulated save failure"):
                export_minizero_torchscript(self.model, self.path)

        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(torch.jit.load(self.path).get_action_size(), 82)


@unittest.skipUnless(
    os.environ.get("XIAQI_TEST_MINIZERO_SELFPLAY") == "1",
    "Set XIAQI_TEST_MINIZERO_SELFPLAY=1 in the CUDA dev container.",
)
class MiniZeroSelfPlayTest(unittest.TestCase):
    def test_collects_one_game_with_the_exported_model(self) -> None:
        self.assertEqual(sys.platform, "linux", "Use the Linux dev container.")
        self.assertTrue(torch.cuda.is_available(), "MiniZero self-play requires CUDA.")
        database_url = os.environ.get("XIAQI_TEST_DATABASE_URL")
        if not database_url:
            self.fail("Set XIAQI_TEST_DATABASE_URL to a PostgreSQL test database.")

        import psycopg
        from psycopg import sql
        from psycopg.conninfo import conninfo_to_dict, make_conninfo

        engine = _REPO_ROOT / "build/minizero/go/minizero_go"
        self.assertTrue(
            engine.is_file() and engine.with_suffix(".commit").is_file(),
            "Build the engine and commit file with scripts/build_minizero.sh.",
        )
        schema = f"torchscript_test_{uuid4().hex}"
        schema_identifier = sql.Identifier(schema)
        options = conninfo_to_dict(database_url).get("options", "")
        test_url = make_conninfo(database_url, options=f"{options} -c search_path={schema}".lstrip())
        connection = self.enterContext(psycopg.connect(test_url, autocommit=True))
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema_identifier))
        self.addCleanup(connection.execute, sql.SQL("DROP SCHEMA {} CASCADE").format(schema_identifier))
        schema_path = _REPO_ROOT / "src/data/postgresql_store/schema.sql"
        connection.execute(schema_path.read_text(encoding="utf-8"))

        directory = self.enterContext(TemporaryDirectory(prefix="xiaqi-minizero-export-"))
        model_path = Path(directory) / "tiny_go_9x9.pt"
        export_minizero_torchscript(_tiny_model(), model_path)
        pool_id = "torchscript-test"
        command = [
            sys.executable, "-m", "scripts.collect_selfplay",
            "--pool-id", pool_id, "--games", "1",
            "--engine", str(engine),
            "--config", "configs/minizero/go_9x9_az.cfg",
            "--model", str(model_path),
            "--conf-str", (
                "program_quiet=true:program_auto_seed=false:zero_num_threads=1:"
                "zero_num_parallel_games=1:actor_num_simulation=2:zero_disable_resign_ratio=1"
            ),
        ]
        with subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            env={**os.environ, "XIAQI_DATABASE_URL": test_url},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        ) as process:
            try:
                output, _ = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                self.fail(f"Collection timed out after 120 seconds.\n{output}")
        self.assertEqual(process.returncode, 0, output)

        rows = connection.execute(
            "SELECT games.ingest_seq, games.record_text, source_runs.model_sha256 "
            "FROM games JOIN source_runs USING (pool_id, source_run_id) "
            "WHERE pool_id = %s",
            (pool_id,),
        ).fetchall()
        self.assertEqual(len(rows), 1, output)
        ingest_seq, record_text, model_sha256 = rows[0]
        self.assertEqual(ingest_seq, 1)
        record = parse_record(record_text)
        self.assertTrue(record.is_complete)
        self.assertEqual(record.board_size, 9)
        self.assertEqual(record.komi, 7.0)
        self.assertEqual(record.model_name, model_path.name)
        with model_path.open("rb") as model_file:
            self.assertEqual(model_sha256, hashlib.file_digest(model_file, "sha256").hexdigest())


if __name__ == "__main__":
    unittest.main()
