import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from .check_minizero import CONFIG_PATH, PASS, check_go_environment, main


class CheckCommandTest(unittest.TestCase):
    def test_reports_how_to_build_missing_binding(self) -> None:
        with patch.dict(sys.modules, {"minizero_py": None}):
            with self.assertRaisesRegex(SystemExit, r"build_minizero.sh --python"):
                main()

    def test_stops_on_config_failure(self) -> None:
        binding = Mock()
        binding.load_config_file.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot load"):
            check_go_environment(binding)
        binding.load_config_file.assert_called_once_with(str(CONFIG_PATH))
        binding.Environment.assert_not_called()


class MiniZeroBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import minizero_py
        except ModuleNotFoundError as error:
            if error.name != "minizero_py":
                raise
            raise unittest.SkipTest(
                "Build minizero_py in dev and set PYTHONPATH."
            ) from error
        cls.binding = minizero_py

    def test_check_runs_to_completion(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            main()
        self.assertIn("MiniZero Go environment check passed.", output.getvalue())

    def test_project_komi_is_applied(self) -> None:
        self.assertTrue(self.binding.load_config_file(str(CONFIG_PATH)))
        env = self.binding.Environment()

        # The empty points form one region touching both colors. Black has
        # eight stones and White has one, so 7 komi makes the position a draw.
        for black_action in range(8):
            self.assertTrue(env.act(black_action))
            self.assertTrue(env.act(PASS - 1 if black_action == 0 else PASS))
        self.assertFalse(env.is_terminal())
        self.assertTrue(env.act(PASS))
        self.assertTrue(env.is_terminal())
        self.assertEqual(env.eval_score(), 0.0)


if __name__ == "__main__":
    unittest.main()
