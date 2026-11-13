"""Activation functions and their derivatives.

Each activation is a pair: the function itself and its derivative with
respect to its input. The network uses ReLU in the hidden layers and
softmax on the output layer; sigmoid and tanh are included so the
network can be built with either of them instead.

Why these exist: a neural network is just a stack of linear layers
(matrix multiply + bias). Without a non-linear activation between them,
the whole stack collapses into a single linear map and could never
learn curved decision boundaries. The activation is what gives the
network its power; its derivative is what makes backpropagation work.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, x). Cheap, and its gradient is 1 for
    positive inputs, so it does not shrink the gradient the way sigmoid
    does (no vanishing-gradient problem for deep stacks)."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where the input was positive, 0 elsewhere."""
    return (x > 0).astype(x.dtype)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: squashes any input into (0, 1). Useful when an output
    should be read as a probability, but its gradient is at most 0.25,
    so stacking many sigmoids makes gradients vanish."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid, written in terms of sigmoid(x) itself:
    s * (1 - s)."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent: squashes inputs into (-1, 1). Zero-centred,
    unlike sigmoid, so its outputs average around 0, which usually
    trains a bit better than sigmoid in hidden layers."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(x)^2."""
    return 1.0 - np.tanh(x) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax: turns a row of raw scores (logits) into a probability
    distribution — every row sums to 1. Used on the output layer so the
    network's answer can be read as "probability of each class".
    The max is subtracted first for numerical stability: it does not
    change the result, but it stops exp() overflowing on large logits."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Registry so the network can be built by name, e.g. activation="tanh".
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
}
