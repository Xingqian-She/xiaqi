"""在终端与 MiniZero 围棋模型对弈。"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from pickle import UnpicklingError
from tempfile import TemporaryDirectory

import torch

from src.control.config_io.config_io import load_training_config
from src.control.selfplay_collector.selfplay_collector import (
    _validate_model,
    training_overrides,
)
from src.core.model.transformer.transformer import Transformer
from src.data_types.go_rules_config.go_rules_config import GoRulesConfig
from src.data_types.training_config.training_config import TrainingConfig
from src.integrations.minizero.torchscript_export import export_minizero_torchscript

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RULES = GoRulesConfig()
_SIMULATIONS = 400
_COLORS = {"B": "黑", "W": "白"}


class _GtpRejection(RuntimeError):
    """MiniZero 返回了 GTP 错误响应。"""


def _gtp(engine: subprocess.Popen[str], command: str) -> str:
    assert engine.stdin is not None and engine.stdout is not None
    engine.stdin.write(command + "\n")
    engine.stdin.flush()

    while (header := engine.stdout.readline()) and not header.strip():
        pass
    if header and header[0] not in "=?":
        raise RuntimeError(f"MiniZero 返回了无效的 GTP 响应 {header.rstrip()!r}")
    if header:
        lines = [header[1:].strip()]
        for line in engine.stdout:
            if not line.strip():
                response = "\n".join(lines).strip("\n")
                if header[0] == "?":
                    raise _GtpRejection(response)
                return response
            lines.append(line.rstrip("\r\n"))
    raise RuntimeError(f"MiniZero 在回应 {command!r} 时退出。")


def _play_game(engine: subprocess.Popen[str], rules: GoRulesConfig, human: str) -> None:
    size = rules.board_size
    columns = "ABCDEFGHJKLMNOPQRST"[:size]
    coordinate = re.compile(rf"[{columns}]([1-9][0-9]?)")
    # MiniZero also ends the game once the move count exceeds twice the board area.
    move_limit = 2 * size * size + 1
    turn = "B"
    moves = consecutive_passes = 0
    print(f"{size}×{size}，贴 {rules.komi:g} 目。每手 MCTS {_SIMULATIONS} 次。")
    print(f"你执{_COLORS[human]}。输入坐标（如 D4）、pass 或 quit。")
    print(_gtp(engine, "showboard"))

    while True:
        if turn == human:
            try:
                move = input(f"你（{_COLORS[turn]}）> ").strip().upper()
            except EOFError:
                return
            if move == "QUIT":
                return
            point = coordinate.fullmatch(move)
            if move != "PASS" and (point is None or int(point[1]) > size):
                print(f"请输入棋盘坐标（{columns[0]}–{columns[-1]}，1–{size}）、pass 或 quit。")
                continue
            try:
                _gtp(engine, f"play {turn} {move}")
            except _GtpRejection as error:
                print(f"这手不能下。{error}")
                continue
        else:
            print(f"AI（{_COLORS[turn]}）思考中…", flush=True)
            move = _gtp(engine, f"genmove {turn}").upper()
            print(f"AI（{_COLORS[turn]}）走 {move}。")

        moves += 1
        consecutive_passes = consecutive_passes + 1 if move == "PASS" else 0
        finished = consecutive_passes == 2 or moves == move_limit
        if finished or turn != human:
            print(_gtp(engine, "showboard"))
        if finished:
            score = float(_gtp(engine, "final_score"))
            result = {1.0: "黑胜", -1.0: "白胜", 0.0: "和棋"}[score]
            print(f"终局，{result}。")
            return
        turn = "W" if turn == "B" else "B"


def _console_command(model_path: Path, config: TrainingConfig | None) -> list[str]:
    engine = config.selfplay.engine_path if config else _REPO_ROOT / "build/minizero/go/minizero_go"
    base_config = (
        config.selfplay.config_path if config else _REPO_ROOT / "runs/go-9x9-az/minizero_base.cfg"
    )
    if not engine.is_file():
        raise FileNotFoundError(f"找不到 MiniZero 引擎 {engine}。请先运行构建脚本。")
    if not base_config.is_file():
        raise FileNotFoundError(f"找不到 MiniZero 配置 {base_config}。")

    model_name = str(model_path)
    if model_name != model_name.strip() or any(char in model_name for char in ":#\r\n"):
        raise ValueError("MiniZero 模型路径不能含冒号、#、换行或首尾空白。")
    overrides = (
        training_overrides(config)
        if config
        else {
            "env_board_size": _DEFAULT_RULES.board_size,
            "env_go_komi": _DEFAULT_RULES.komi,
            "env_go_ko_rule": _DEFAULT_RULES.ko_rule,
        }
    )
    overrides.update(
        nn_file_name=model_name,
        actor_num_simulation=_SIMULATIONS,
        actor_mcts_think_time_limit=0,
        actor_select_action_by_count=True,
        actor_select_action_by_softmax_count=False,
        actor_use_dirichlet_noise=False,
        actor_use_gumbel=False,
        actor_use_gumbel_noise=False,
        zero_disable_resign_ratio=1.0,
        program_quiet=True,
        program_use_color_message=True,
    )
    conf_str = ":".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in overrides.items()
    )
    return [str(engine), "-mode", "console", "-conf_file", str(base_config), "-conf_str", conf_str]


def _run_console(model_path: Path, config: TrainingConfig | None, human: str) -> None:
    command = _console_command(model_path, config)
    rules = config.go if config else _DEFAULT_RULES
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8"
    ) as engine:
        try:
            _play_game(engine, rules, human)
        finally:
            # MiniZero's console exits on quit without sending a GTP response.
            try:
                engine.communicate(input="quit\n", timeout=5)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.communicate()
        if engine.returncode:
            raise RuntimeError(f"MiniZero 退出，状态码为 {engine.returncode}。")


def _play_checkpoint(checkpoint: Path, config: TrainingConfig, human: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"无法读取检查点 {checkpoint} 的权重。")
    key = "ema_model" if "ema_model" in state else "model"
    weights = state.get(key)
    if not isinstance(weights, dict):
        raise ValueError(f"检查点 {checkpoint} 缺少有效的 {key} 权重。")
    model = Transformer.from_config(config.model)
    model.load_state_dict(weights, strict=True)
    del state, weights
    with TemporaryDirectory(prefix="xiaqi-play-", dir="/dev/shm") as directory:
        model_path = Path(directory) / "model.pt"
        export_minizero_torchscript(model, model_path)
        del model
        label = "EMA 权重" if key == "ema_model" else "训练权重"
        print(f"已加载 {label} {checkpoint.name}。")
        _run_console(model_path, config, human)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="训练检查点，优先使用其中的 EMA 权重。")
    source.add_argument("--model", type=Path, help="MiniZero 围棋 AlphaZero TorchScript 文件。")
    parser.add_argument("--config", type=Path, help="训练 TOML。检查点默认使用同一实验的配置。")
    parser.add_argument("--color", choices=("black", "white"), default="black", help="你执黑或执白。")
    args = parser.parse_args(argv)

    human = "B" if args.color == "black" else "W"
    try:
        path = (args.checkpoint or args.model).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"请选择一个权重文件 {path}。")
        config_path = args.config
        if args.checkpoint and config_path is None:
            run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
            config_path = run_dir / "config.toml"
            if not config_path.is_file():
                raise ValueError(f"找不到运行配置 {config_path}。请用 --config 指定训练 TOML。")
        config = load_training_config(config_path) if config_path is not None else None
        if args.checkpoint:
            assert config is not None
            _play_checkpoint(path, config, human)
        else:
            rules = config.go if config else _DEFAULT_RULES
            _validate_model(path, rules.board_size)
            print(f"已加载 MiniZero 权重 {path.name}。")
            _run_console(path, config, human)
    except (OSError, RuntimeError, ValueError, UnpicklingError) as error:
        print(f"对弈失败。{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n对弈已退出。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
