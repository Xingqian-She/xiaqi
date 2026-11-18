"""Check the GPU toolchain and torch.compile forward/backward correctness."""

from __future__ import annotations

import platform
import shutil
import subprocess

import torch


def workload(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sin(left @ right)


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")

    for tool in ("g++", "ninja", "nvcc"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Missing compiler tool: {tool}. Rebuild the development image.")
    subprocess.run(["nvcc", "--version"], check=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run inside the GPU-enabled dev container.")
    if not torch.backends.cudnn.is_available():
        raise SystemExit("cuDNN is unavailable.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    left = (torch.randn(32, 64, device="cuda") * 0.1).requires_grad_()
    right = (torch.randn(64, 32, device="cuda") * 0.1).requires_grad_()
    compiled_left = left.detach().clone().requires_grad_()
    compiled_right = right.detach().clone().requires_grad_()

    expected = workload(left, right)
    expected.square().mean().backward()
    torch.cuda.synchronize()
    print("Eager GPU forward/backward passed.")

    compiled = torch.compile(workload, backend="inductor", fullgraph=True)
    actual = compiled(compiled_left, compiled_right)
    actual.square().mean().backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    for eager_input, compiled_input in ((left, compiled_left), (right, compiled_right)):
        if eager_input.grad is None or compiled_input.grad is None:
            raise RuntimeError("Backward did not produce input gradients.")
        torch.testing.assert_close(
            compiled_input.grad, eager_input.grad, rtol=1e-4, atol=1e-5
        )
        if not torch.isfinite(compiled_input.grad).all().item():
            raise RuntimeError("Compiled gradients contain non-finite values.")
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("Compiled output contains non-finite values.")

    print("torch.compile (Inductor) forward/backward matches eager.")
    print("xiaqi GPU development environment check passed.")


if __name__ == "__main__":
    main()
