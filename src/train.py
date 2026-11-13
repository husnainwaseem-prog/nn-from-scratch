"""Training: mini-batch gradient descent with momentum.

One epoch = one pass over the training set, split into shuffled
mini-batches (128 examples each). For each batch: forward pass,
backpropagation, one momentum-SGD step. After each epoch the loss and
accuracy are measured on the full training set and on the test set —
the training curve shows whether the network is actually learning
(loss falling) and whether it generalises (test accuracy tracking
train accuracy instead of diverging from it).

History is saved to results/training_history.json and plotted to
results/plots/training_curves.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .data import load_data
from .network import NeuralNetwork, one_hot
from .optimizer import SGDMomentum

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HISTORY_FILE = RESULTS_DIR / "training_history.json"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULTS = {
    "layer_sizes": [784, 128, 64, 10],
    "activation": "relu",
    "lr": 0.05,
    "momentum": 0.9,
    "batch_size": 128,
    "epochs": 10,
    "seed": 42,
}


def train(
    layer_sizes: list[int] | None = None,
    activation: str = "relu",
    lr: float = 0.05,
    momentum: float = 0.9,
    batch_size: int = 128,
    epochs: int = 10,
    seed: int = 42,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[NeuralNetwork, dict]:
    """Train a network from scratch and return (network, history).

    History records, per epoch: train/test loss and accuracy, plus
    the wall-clock training time. The network itself is returned so
    callers (tests, compare) can evaluate it without retraining.
    """
    layer_sizes = layer_sizes or list(DEFAULTS["layer_sizes"])
    if X_train is None:
        X_train, y_train, X_test, y_test = load_data()
    assert y_train is not None and X_test is not None and y_test is not None

    n_classes = layer_sizes[-1]
    net = NeuralNetwork(layer_sizes, activation=activation, seed=seed)
    opt = SGDMomentum(lr=lr, momentum=momentum)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    steps_per_epoch = int(np.ceil(n / batch_size))

    history: dict = {
        "config": {
            "layer_sizes": layer_sizes,
            "activation": activation,
            "lr": lr,
            "momentum": momentum,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
        },
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch_times_s": [],
    }

    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        perm = rng.permutation(n)
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            X_batch, y_batch = X_train[idx], y_train[idx]
            net.forward(X_batch)
            grads = net.backward(X_batch, one_hot(y_batch, n_classes))
            opt.update(net.params, grads)
        history["epoch_times_s"].append(round(time.time() - epoch_start, 2))
        history["train_loss"].append(round(net.loss(X_train, y_train), 4))
        history["train_acc"].append(round(net.accuracy(X_train, y_train), 4))
        history["test_loss"].append(round(net.loss(X_test, y_test), 4))
        history["test_acc"].append(round(net.accuracy(X_test, y_test), 4))
        if verbose:
            print(
                f"epoch {epoch:2d}/{epochs} | train loss {history['train_loss'][-1]:.4f} "
                f"acc {history['train_acc'][-1]:.4f} | test loss {history['test_loss'][-1]:.4f} "
                f"acc {history['test_acc'][-1]:.4f} | {history['epoch_times_s'][-1]:.1f}s"
            )
    history["train_time_s"] = round(time.time() - start, 2)
    history["final_test_acc"] = history["test_acc"][-1]
    history["final_train_acc"] = history["train_acc"][-1]
    return net, history


def save_history(history: dict) -> Path:
    """Write the history JSON to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))
    return HISTORY_FILE


def plot_training_curves(history: dict, out_path: Path | None = None) -> Path:
    """Two-panel plot: loss per epoch and accuracy per epoch, train vs
    test. Saved to results/plots/training_curves.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (PLOTS_DIR / "training_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["test_loss"], marker="s", label="test")
    axes[0].set_title("Cross-entropy loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["test_acc"], marker="s", label="test")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_training(**kwargs) -> tuple[NeuralNetwork, dict]:
    """Train with the defaults, save history + curves, print a summary."""
    net, history = train(**kwargs)
    save_history(history)
    plot_path = plot_training_curves(history)
    print(f"\nDone in {history['train_time_s']}s — final test accuracy {history['final_test_acc']:.4f}")
    print(f"History: {HISTORY_FILE}\nCurves: {plot_path}")
    return net, history
