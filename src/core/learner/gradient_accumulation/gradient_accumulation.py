"""用等长 mini batch 累积一次策略与价值网络更新的梯度。"""

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ....data_types.training_samples.training_samples import TrainingSamples


def train_accumulated_batch(
    model: nn.Module,
    optimizer: Optimizer,
    batch: TrainingSamples,
    *,
    device: torch.device,
    mini_batch_size: int,
    value_loss_weight: float,
) -> tuple[float, float, float]:
    """更新一次参数，返回完整批次的策略、价值和总损失。"""
    batch_size = len(batch.features)
    if not 1 <= mini_batch_size <= batch_size or batch_size % mini_batch_size:
        raise ValueError(
            "mini_batch_size must divide the batch size and not exceed it."
        )
    mini_batch_count = batch_size // mini_batch_size

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum: Tensor | None = None
    try:
        for start in range(0, batch_size, mini_batch_size):
            rows = slice(start, start + mini_batch_size)
            board = torch.from_numpy(batch.features[rows]).to(device)
            policy = torch.from_numpy(batch.policy[rows]).to(device)
            value = torch.from_numpy(batch.value[rows]).to(device)
            logits, value_pred = model(board)
            if logits.shape != policy.shape:
                raise ValueError(
                    f"Expected policy logits shape {tuple(policy.shape)}, got {tuple(logits.shape)}."
                )
            if value_pred.shape != value.shape:
                raise ValueError(
                    f"Expected value prediction shape {tuple(value.shape)}, "
                    f"got {tuple(value_pred.shape)}."
                )

            policy_loss = -(policy * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            value_loss = (value_pred - value).square().mean()
            total_loss = policy_loss + value_loss_weight * value_loss
            (total_loss / mini_batch_count).backward()

            losses = torch.stack(
                (policy_loss.detach(), value_loss.detach(), total_loss.detach())
            )
            if loss_sum is None:
                loss_sum = losses
            else:
                loss_sum.add_(losses)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise

    optimizer.step()
    assert loss_sum is not None
    policy_mean, value_mean, total_mean = (loss_sum / mini_batch_count).tolist()
    return policy_mean, value_mean, total_mean
