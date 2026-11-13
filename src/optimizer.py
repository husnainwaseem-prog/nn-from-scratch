"""SGD optimizer with momentum.

Plain gradient descent steps directly downhill:  w -= lr * grad.
The problem is that it zig-zags in narrow valleys and crawls on flat
plateaus. Momentum fixes both: it keeps a running average (velocity)
of past gradients, so consistent directions build up speed while
opposing directions cancel out — like a heavy ball rolling downhill.

Update rule, per parameter:
    velocity = momentum * velocity - lr * grad
    param   += velocity
"""

from __future__ import annotations

import numpy as np


class SGDMomentum:
    """Stochastic gradient descent with momentum.

    Args:
        lr: learning rate — how big each step is.
        momentum: how much of the previous velocity to keep (0.9 is the
            standard choice: the velocity is roughly a weighted average
            of the last ~10 gradients).
    """

    def __init__(self, lr: float = 0.05, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.velocities: dict[str, np.ndarray] = {}

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Apply one optimizer step, updating `params` in place.

        A separate velocity array is kept for every parameter array,
        created (as zeros) the first time that parameter is seen.
        """
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            v = self.velocities[key]
            v = self.momentum * v - self.lr * grads[key]
            self.velocities[key] = v
            params[key] += v
