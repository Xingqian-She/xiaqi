"""为推理维护训练参数的指数移动平均。"""

from copy import deepcopy
from numbers import Real

import torch
from torch import nn


class ExponentialMovingAverage:
    """保存与训练模型使用同一设备的冻结推理副本。"""

    def __init__(self, source_model: nn.Module, tau: float) -> None:
        if not isinstance(tau, Real) or isinstance(tau, bool) or not 0 < tau <= 1:
            raise ValueError("tau must be in (0, 1].")
        self._source_model = source_model
        self._tau = float(tau)
        self.model = deepcopy(source_model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self) -> None:
        """在一次成功的优化器更新后平均参数并直接同步缓冲区。"""
        for average, current in zip(
            self.model.parameters(), self._source_model.parameters(), strict=True
        ):
            if self._tau < 1 and (average.is_floating_point() or average.is_complex()):
                average.lerp_(current, self._tau)
            else:
                average.copy_(current)
        for average, current in zip(
            self.model.buffers(), self._source_model.buffers(), strict=True
        ):
            average.copy_(current)
