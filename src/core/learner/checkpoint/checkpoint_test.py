import unittest
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn

from ..optimizer.optimizer import prepare_optimizer
from .checkpoint import load_checkpoint, save_checkpoint


class CheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(self.enterContext(TemporaryDirectory())) / "learner.pt"
        self.model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.make_optimizer = partial(torch.optim.AdamW, lr=0.01, weight_decay=0.02)
        self.optimizer, self.normalize_optimizer_state = prepare_optimizer(
            self.model, self.make_optimizer
        )

    def test_checkpoint_restores_weights_optimizer_and_step(self) -> None:
        self.model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        save_checkpoint(self.path, model=self.model, optimizer=self.optimizer, step=1)
        original = torch.load(self.path, weights_only=True)

        resumed_model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        resumed_optimizer, normalize_state = prepare_optimizer(
            resumed_model, partial(torch.optim.AdamW, lr=0.5)
        )
        step = load_checkpoint(
            str(self.path),
            model=resumed_model,
            optimizer=resumed_optimizer,
            normalize_optimizer_state=normalize_state,
        )

        self.assertEqual(set(original), {"model", "optimizer", "step"})
        self.assertEqual(step, 1)
        torch.testing.assert_close(
            resumed_model.state_dict(), original["model"], rtol=0, atol=0
        )
        restored = resumed_optimizer.state_dict()
        self.assertEqual(
            restored["param_groups"], original["optimizer"]["param_groups"]
        )
        self.assertTrue(original["optimizer"]["state"])
        torch.testing.assert_close(
            restored["state"], original["optimizer"]["state"], rtol=0, atol=0
        )

    def test_checkpoint_restores_legacy_optimizer_parameter_order(self) -> None:
        legacy_model = deepcopy(self.model)
        legacy_optimizer = self.make_optimizer(legacy_model.parameters())
        for index, param in enumerate(legacy_model.parameters(), start=1):
            param.grad = torch.full_like(param, index)
        legacy_optimizer.step()
        save_checkpoint(
            self.path, model=legacy_model, optimizer=legacy_optimizer, step=1
        )

        step = load_checkpoint(
            self.path,
            model=self.model,
            optimizer=self.optimizer,
            normalize_optimizer_state=self.normalize_optimizer_state,
        )

        self.assertEqual(step, 1)
        self.assertEqual(
            [
                (group["lr"], group["weight_decay"])
                for group in self.optimizer.param_groups
            ],
            [(0.01, 0.02), (0.01, 0.0)],
        )
        for old_param, new_param in zip(
            legacy_model.parameters(), self.model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                self.optimizer.state[new_param], legacy_optimizer.state[old_param]
            )

    def test_legacy_checkpoint_initializes_ema_from_restored_model(self) -> None:
        ema_model = deepcopy(self.model).eval().requires_grad_(False)
        self.model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        save_checkpoint(self.path, model=self.model, optimizer=self.optimizer, step=1)
        saved_model = deepcopy(self.model.state_dict())
        self.assertFalse(torch.equal(ema_model[1].bias, saved_model["1.bias"]))

        step = load_checkpoint(
            self.path,
            model=self.model,
            optimizer=self.optimizer,
            normalize_optimizer_state=self.normalize_optimizer_state,
            ema_model=ema_model,
        )

        self.assertEqual(step, 1)
        torch.testing.assert_close(ema_model.state_dict(), saved_model, rtol=0, atol=0)

    def test_invalid_ema_state_rolls_back_model_optimizer_and_ema(self) -> None:
        self.model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        ema_model = deepcopy(self.model).eval().requires_grad_(False)
        before_model = deepcopy(self.model.state_dict())
        before_optimizer = deepcopy(self.optimizer.state_dict())
        before_ema = deepcopy(ema_model.state_dict())
        invalid_ema = {name: value + 2 for name, value in before_ema.items()}
        invalid_ema["1.bias"] = torch.zeros(3)
        torch.save(
            {
                "model": {name: value + 1 for name, value in before_model.items()},
                "optimizer": {
                    "param_groups": [
                        {**group, "lr": 0.5}
                        for group in before_optimizer["param_groups"]
                    ],
                    "state": {},
                },
                "ema_model": invalid_ema,
                "step": 10,
            },
            self.path,
        )

        with self.assertRaisesRegex(RuntimeError, "size mismatch for 1.bias"):
            load_checkpoint(
                self.path,
                model=self.model,
                optimizer=self.optimizer,
                normalize_optimizer_state=self.normalize_optimizer_state,
                ema_model=ema_model,
            )

        torch.testing.assert_close(
            self.model.state_dict(), before_model, rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.optimizer.state_dict(), before_optimizer, rtol=0, atol=0
        )
        torch.testing.assert_close(ema_model.state_dict(), before_ema, rtol=0, atol=0)

    def test_failed_checkpoint_write_preserves_previous_file(self) -> None:
        def fail_after_write(state, file) -> None:
            file.write(b"incomplete checkpoint")
            raise OSError("simulated write failure")

        save_checkpoint(self.path, model=self.model, optimizer=self.optimizer, step=1)
        previous = self.path.read_bytes()
        with patch.object(torch, "save", side_effect=fail_after_write):
            with self.assertRaisesRegex(OSError, "simulated write failure"):
                save_checkpoint(
                    self.path, model=self.model, optimizer=self.optimizer, step=2
                )
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_incompatible_optimizer_checkpoint_preserves_training_state(self) -> None:
        self.model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        before_model = deepcopy(self.model.state_dict())
        before_optimizer = deepcopy(self.optimizer.state_dict())
        invalid = {
            "model": {name: value + 1 for name, value in before_model.items()},
            "optimizer": {**before_optimizer, "param_groups": []},
            "step": 10,
        }
        torch.save(invalid, self.path)

        def mutate_before_validation(optimizer, state) -> None:
            if not state["param_groups"]:
                optimizer.param_groups[0]["lr"] = 0.5
                optimizer.state.clear()

        with self.optimizer.register_load_state_dict_pre_hook(mutate_before_validation):
            with self.assertRaisesRegex(ValueError, "parameter groups"):
                load_checkpoint(
                    self.path,
                    model=self.model,
                    optimizer=self.optimizer,
                    normalize_optimizer_state=self.normalize_optimizer_state,
                )

        torch.testing.assert_close(
            self.model.state_dict(), before_model, rtol=0, atol=0
        )
        restored = self.optimizer.state_dict()
        self.assertEqual(restored["param_groups"], before_optimizer["param_groups"])
        torch.testing.assert_close(
            restored["state"], before_optimizer["state"], rtol=0, atol=0
        )

    def test_rejects_invalid_step(self) -> None:
        for step in (-1, True, 1.0):
            with self.subTest(step=step):
                torch.save({"model": {}, "optimizer": {}, "step": step}, self.path)
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    load_checkpoint(
                        self.path,
                        model=self.model,
                        optimizer=self.optimizer,
                        normalize_optimizer_state=self.normalize_optimizer_state,
                    )


if __name__ == "__main__":
    unittest.main()
