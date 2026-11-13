"""Gradient checking: verify backpropagation numerically.

Backpropagation is easy to get subtly wrong (a transposed matrix, a
missing 1/m, a wrong derivative) and a network will still "train" —
just badly. Gradient checking catches those bugs before training
starts.

Method: for each weight, compute the gradient two ways and compare.
1. Analytic — from backpropagation.
2. Numerical — central difference: nudge the weight by +eps and -eps,
   measure the loss both ways, and take (loss+ - loss-) / (2*eps).
   This approximates the true derivative without using backprop at all.

Agreement is measured with the relative error
    |numerical - analytic| / (|numerical| + |analytic|)
per weight; the check passes if the maximum over all weights is below
1e-6. It runs on a tiny network (4 -> 5 -> 3, 5 samples) because the
numerical pass needs two full forward passes per weight, which is far
too slow on the real 784-128-64-10 network.
"""

from __future__ import annotations

import numpy as np

from .network import NeuralNetwork, one_hot


def numerical_gradients(
    net: NeuralNetwork, X: np.ndarray, y: np.ndarray, eps: float = 1e-5
) -> dict[str, np.ndarray]:
    """Central-difference gradients for every parameter."""
    grads: dict[str, np.ndarray] = {}
    for key, param in net.params.items():
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            loss_plus = net.loss(X, y)
            param[idx] = original - eps
            loss_minus = net.loss(X, y)
            param[idx] = original
            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()
        grads[key] = num_grad
    return grads


def gradient_check(
    layer_sizes: list[int] | None = None,
    n_samples: int = 5,
    eps: float = 1e-5,
    threshold: float = 1e-6,
    seed: int = 7,
    verbose: bool = True,
) -> dict:
    """Run the check and return a report dict.

    Report keys: max_relative_error, per-parameter max errors,
    n_params, passed.
    """
    layer_sizes = layer_sizes or [4, 5, 3]
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, layer_sizes[0]))
    y = rng.integers(0, layer_sizes[-1], size=n_samples)

    net = NeuralNetwork(layer_sizes, activation="relu", seed=seed)
    net.forward(X)
    analytic = net.backward(X, one_hot(y, layer_sizes[-1]))
    numerical = numerical_gradients(net, X, y, eps=eps)

    per_param: dict[str, float] = {}
    max_error = 0.0
    for key in analytic:
        diff = np.abs(numerical[key] - analytic[key])
        denom = np.abs(numerical[key]) + np.abs(analytic[key])
        rel = np.max(diff / np.maximum(denom, 1e-12))
        per_param[key] = float(rel)
        max_error = max(max_error, float(rel))

    passed = max_error < threshold
    report = {
        "layer_sizes": layer_sizes,
        "n_samples": n_samples,
        "eps": eps,
        "threshold": threshold,
        "n_params": int(sum(p.size for p in net.params.values())),
        "per_param_max_relative_error": {k: round(v, 10) for k, v in per_param.items()},
        "max_relative_error": max_error,
        "passed": passed,
    }
    if verbose:
        for key, err in per_param.items():
            print(f"  {key}: max relative error {err:.2e}")
        print(f"Max relative error: {max_error:.2e}  (threshold {threshold:.0e}) -> {'PASS' if passed else 'FAIL'}")
    return report
