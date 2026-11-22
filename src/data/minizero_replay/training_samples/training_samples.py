from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)
class TrainingSamples:
    """Float32 arrays with aligned rows and shapes (N, C, H, W), (N, A), (N, 1)."""

    features: NDArray[np.float32]
    policy: NDArray[np.float32]
    value: NDArray[np.float32]
