import unittest
from functools import partial
from unittest.mock import Mock

import torch
from torch import nn

from ...model.transformer.transformer import Transformer
from .optimizer import prepare_optimizer


class OptimizerTest(unittest.TestCase):
    def test_weight_decay_skips_cls_token_and_small_parameters(self) -> None:
        model = Transformer(
            board_size=9,
            in_channels=18,
            model_dim=16,
            num_heads=2,
            ffn_hidden_dim=24,
            num_layers=1,
            value_hidden_dim=8,
        )
        layer_scale = nn.Parameter(torch.full((16,), 1e-5))
        scalar = nn.Parameter(torch.tensor(1.0))
        model.register_parameter("layer_scale", layer_scale)
        model.register_parameter("scalar", scalar)
        optimizer, _ = prepare_optimizer(
            model, partial(torch.optim.AdamW, lr=0.01, weight_decay=0.02)
        )

        no_decay = (
            model.cls_token,
            layer_scale,
            scalar,
            model.encoder.blocks[0].attention_norm.weight,
        )
        decay = (model.stem.conv.weight, model.point_head.weight)
        before = {param: param.detach().clone() for param in (*no_decay, *decay)}
        for param in before:
            param.grad = torch.zeros_like(param)
        optimizer.step()

        for param in no_decay:
            torch.testing.assert_close(param, before[param], rtol=0, atol=0)
        for param in decay:
            torch.testing.assert_close(param, before[param] * (1 - 0.01 * 0.02))
        for param in no_decay:
            param.grad = torch.ones_like(param)
        optimizer.step()
        for param in no_decay:
            self.assertFalse(torch.equal(param, before[param]))

    def test_splits_factory_groups_preserving_options_and_names(self) -> None:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2))
        parameters = dict(model.named_parameters())
        first_names = ("1.bias", "0.weight", "0.bias")
        second_names = ("2.bias", "2.weight", "1.weight")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [(name, parameters[name]) for name in first_names],
                    "lr": 0.01,
                    "weight_decay": 0.02,
                },
                {
                    "params": [(name, parameters[name]) for name in second_names],
                    "lr": 0.04,
                    "weight_decay": 0.0,
                },
            ]
        )
        make_optimizer = Mock(return_value=optimizer)

        prepared, _ = prepare_optimizer(model, make_optimizer)

        self.assertIs(prepared, optimizer)
        make_optimizer.assert_called_once()
        factory_params = make_optimizer.call_args.args[0]
        self.assertEqual(
            list(map(id, factory_params)), list(map(id, model.parameters()))
        )
        self.assertEqual(
            [
                (group["param_names"], group["lr"], group["weight_decay"])
                for group in optimizer.param_groups
            ],
            [
                (["0.weight"], 0.01, 0.02),
                (["1.bias", "0.bias"], 0.01, 0.0),
                (["2.weight", "1.weight"], 0.04, 0.0),
                (["2.bias"], 0.04, 0.0),
            ],
        )
        for group in optimizer.param_groups:
            for name, param in zip(group["param_names"], group["params"], strict=True):
                self.assertIs(param, parameters[name])

    def test_optimizer_without_weight_decay_keeps_original_group(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Rprop(model.parameters())
        (original_group,) = optimizer.param_groups

        prepared, _ = prepare_optimizer(model, lambda _: optimizer)

        self.assertIs(prepared, optimizer)
        (group,) = optimizer.param_groups
        self.assertIs(group, original_group)
        self.assertNotIn("weight_decay", group)


if __name__ == "__main__":
    unittest.main()
