"""配置优化器的权重衰减分组并转换旧检查点的参数组。"""

from collections.abc import Callable, Iterable
from typing import Any

from torch import nn
from torch.optim import Optimizer


def prepare_optimizer(
    model: nn.Module,
    make_optimizer: Callable[[Iterable[nn.Parameter]], Optimizer],
) -> tuple[Optimizer, Callable[[dict[str, Any]], dict[str, Any]]]:
    """返回分组后的原优化器和用于加载检查点的状态转换函数。"""
    optimizer = make_optimizer(model.parameters())
    no_decay = {
        param
        for name, param in model.named_parameters(remove_duplicate=False)
        if name.rsplit(".", 1)[-1] == "cls_token"
        or (not isinstance(param, nn.UninitializedParameter) and param.ndim <= 1)
    }
    # 掩码按工厂的原组内顺序记录，用于拆分旧检查点中的参数 ID。
    no_decay_masks = [
        ["weight_decay" in group and param in no_decay for param in group["params"]]
        for group in optimizer.param_groups
    ]
    optimizer.param_groups[:] = _split_weight_decay_groups(
        optimizer.param_groups, no_decay_masks
    )

    def normalize_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
        groups = state["param_groups"]
        if len(groups) != len(no_decay_masks):
            return state
        return {
            **state,
            "param_groups": _split_weight_decay_groups(groups, no_decay_masks),
        }

    return optimizer, normalize_optimizer_state


def _split_weight_decay_groups(
    groups: list[dict[str, Any]], no_decay_masks: list[list[bool]]
) -> list[dict[str, Any]]:
    split_groups = []
    for group, mask in zip(groups, no_decay_masks, strict=True):
        params = group["params"]
        if len(params) != len(mask):
            raise ValueError(
                "Checkpoint optimizer parameter group size does not match."
            )
        if not any(mask):
            split_groups.append(group)
            continue

        for skip_decay in (False, True):
            indices = [index for index, skip in enumerate(mask) if skip == skip_decay]
            if not indices:
                continue
            new_group = {**group, "params": [params[index] for index in indices]}
            if "param_names" in group:
                new_group["param_names"] = [
                    group["param_names"][index] for index in indices
                ]
            if skip_decay:
                new_group["weight_decay"] = 0.0
            split_groups.append(new_group)
    return split_groups
