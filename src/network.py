"""Feedforward neural network implemented from scratch in NumPy.

Architecture: an input layer, one or more hidden layers with a chosen
activation (ReLU by default), and an output layer with softmax.
Trained with mini-batch gradient descent and cross-entropy loss.

The maths, in one paragraph: the forward pass pushes a batch of
inputs through the layers, caching every intermediate value. The
loss is cross-entropy between the softmax probabilities and the true
labels. The backward pass uses the chain rule to work out how much
each weight contributed to the loss — for softmax + cross-entropy the
output-layer error has the beautifully simple form
``(probabilities - one_hot_labels) / batch_size`` — and propagates
that error backwards through the layers, multiplying by each
activation's derivative on the way. The optimizer then nudges every
weight against its gradient.

Weights are initialised with He initialisation (variance 2/fan_in) for
ReLU hidden layers, which keeps the signal from shrinking or blowing
up as it passes through the layers, and Xavier initialisation
(variance 1/fan_in) for the softmax output layer.
"""

from __future__ import annotations

import numpy as np

from .activations import ACTIVATIONS, softmax


def cross_entropy_loss(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    """Mean cross-entropy loss over a batch. Probabilities are clipped
    so log(0) can never produce -inf."""
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(y_onehot * np.log(probs), axis=1)))


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer class labels (shape (m,)) to one-hot rows (m, n)."""
    out = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


class NeuralNetwork:
    """A fully-connected feedforward network.

    Args:
        layer_sizes: e.g. [784, 128, 64, 10] — input size, hidden
            sizes, output size (one per class).
        activation: hidden-layer activation name: "relu", "sigmoid"
            or "tanh".
        seed: random seed, so runs are reproducible.
    """

    def __init__(self, layer_sizes: list[int], activation: str = "relu", seed: int = 42) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        self.layer_sizes = list(layer_sizes)
        self.activation, self.activation_derivative = ACTIVATIONS[activation]
        self.rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        self._init_params()
        # Cache of the forward pass, filled by forward() and read by backward().
        self.cache: dict[str, np.ndarray] = {}

    def _init_params(self) -> None:
        for i in range(1, len(self.layer_sizes)):
            fan_in = self.layer_sizes[i - 1]
            fan_out = self.layer_sizes[i]
            is_output = i == len(self.layer_sizes) - 1
            # He for ReLU hidden layers, Xavier for the output layer.
            scale = np.sqrt(1.0 / fan_in) if is_output else np.sqrt(2.0 / fan_in)
            self.params[f"W{i}"] = self.rng.normal(0.0, scale, size=(fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=np.float64)

    @property
    def n_layers(self) -> int:
        """Number of weight layers (hidden + output)."""
        return len(self.layer_sizes) - 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation. Returns softmax probabilities (m, n_classes)
        and caches every layer's pre-activation (Z) and activation (A)."""
        A = X.astype(np.float64)
        self.cache = {"A0": A}
        for i in range(1, self.n_layers + 1):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            self.cache[f"Z{i}"] = Z
            if i == self.n_layers:
                A = softmax(Z)
            else:
                A = self.activation(Z)
            self.cache[f"A{i}"] = A
        return A

    def backward(self, X: np.ndarray, y_onehot: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagation. Assumes forward() was just called on the same
        batch. Returns gradients for every parameter."""
        m = X.shape[0]
        grads: dict[str, np.ndarray] = {}
        # Output layer: softmax + cross-entropy gives dZ = (probs - y) / m.
        dZ = (self.cache[f"A{self.n_layers}"] - y_onehot) / m
        for i in range(self.n_layers, 0, -1):
            A_prev = self.cache[f"A{i - 1}"]
            grads[f"W{i}"] = A_prev.T @ dZ
            grads[f"b{i}"] = np.sum(dZ, axis=0)
            if i > 1:
                dA = dZ @ self.params[f"W{i}"].T
                dZ = dA * self.activation_derivative(self.cache[f"Z{i - 1}"])
        return grads

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy loss of the network on (X, y)."""
        probs = self.forward(X)
        return cross_entropy_loss(probs, one_hot(y, self.layer_sizes[-1]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class index per row (argmax of the softmax output)."""
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of rows predicted correctly."""
        return float(np.mean(self.predict(X) == y))
