"""Comparison: the from-scratch NumPy network vs a framework equivalent.

The point is not to beat PyTorch — it is to show the hand-written
network is correct. The same architecture (784-128-64-10, ReLU),
optimizer (SGD, momentum 0.9), learning rate (0.05), batch size (128),
epochs (10), data split and seed are used on both sides, so any gap
in accuracy or time is a real difference in implementation, not in
setup. Test accuracy and wall-clock training time are recorded for
both, saved to results/comparison.json and plotted to
results/plots/comparison.png.

If PyTorch is not installed, the framework side falls back to
sklearn's MLPClassifier (same shape, same optimizer settings) and the
report says so — no PyTorch numbers are ever invented.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .data import load_data
from .train import train as train_numpy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COMPARISON_FILE = RESULTS_DIR / "comparison.json"
PLOTS_DIR = RESULTS_DIR / "plots"


def train_torch(X_train, y_train, X_test, y_test, epochs=10, lr=0.05, momentum=0.9, batch_size=128, seed=42):
    """Train the same architecture in PyTorch (CPU). Returns
    (test accuracy, training time in seconds)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    n = X_train.shape[0]
    steps = int(np.ceil(n / batch_size))
    gen = torch.Generator().manual_seed(seed)

    start = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(steps):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            opt.zero_grad()
            out = model(X_tr[idx])
            loss = loss_fn(out, y_tr[idx])
            loss.backward()
            opt.step()
    train_time = time.time() - start

    with torch.no_grad():
        preds = model(X_te).argmax(dim=1)
        acc = float((preds == y_te).float().mean())
    return acc, train_time


def train_sklearn_mlp(X_train, y_train, X_test, y_test, epochs=10, lr=0.05, momentum=0.9, batch_size=128, seed=42):
    """Fallback framework equivalent: sklearn's MLPClassifier with the
    same layer sizes and SGD settings. Used only if PyTorch is absent."""
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="sgd",
        learning_rate_init=lr,
        momentum=momentum,
        batch_size=batch_size,
        max_iter=epochs,
        random_state=seed,
    )
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    return float(clf.score(X_test, y_test)), train_time


def plot_comparison(comparison: dict, out_path: Path | None = None) -> Path:
    """Bar chart: test accuracy and training time, NumPy vs framework."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or (PLOTS_DIR / "comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["NumPy\n(from scratch)", f"{comparison['framework_name']}\n(framework)"]
    accs = [comparison["numpy"]["test_acc"], comparison["framework"]["test_acc"]]
    times = [comparison["numpy"]["train_time_s"], comparison["framework"]["train_time_s"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars = axes[0].bar(labels, accs, color=["#2f7f4f", "#3b6ea5"])
    axes[0].set_title("Test accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center")
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(labels, times, color=["#2f7f4f", "#3b6ea5"])
    axes[1].set_title("Training time (seconds)")
    axes[1].set_ylabel("Seconds")
    for bar, v in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + max(times) * 0.02, f"{v:.1f}s", ha="center")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_comparison() -> dict:
    """Train both sides, save comparison.json + plot, print a summary."""
    X_train, y_train, X_test, y_test = load_data()
    n_features = X_train.shape[1]
    layer_sizes = [n_features, 128, 64, 10]

    print("NumPy (from scratch)...")
    _, history = train_numpy(
        layer_sizes=layer_sizes, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, verbose=False
    )
    numpy_result = {"test_acc": round(history["final_test_acc"], 4), "train_time_s": round(history["train_time_s"], 2)}

    try:
        import torch  # noqa: F401

        framework_name = "PyTorch"
        print("PyTorch...")
        acc, t = train_torch(X_train, y_train, X_test, y_test)
    except ImportError:
        framework_name = "sklearn MLPClassifier"
        print("PyTorch not installed — falling back to sklearn MLPClassifier.")
        acc, t = train_sklearn_mlp(X_train, y_train, X_test, y_test)
    framework_result = {"test_acc": round(acc, 4), "train_time_s": round(t, 2)}

    comparison = {
        "config": {
            "layer_sizes": layer_sizes,
            "activation": "relu",
            "lr": 0.05,
            "momentum": 0.9,
            "batch_size": 128,
            "epochs": 10,
            "seed": 42,
        },
        "numpy": numpy_result,
        "framework": framework_result,
        "framework_name": framework_name,
        "test_acc_gap": round(abs(numpy_result["test_acc"] - framework_result["test_acc"]), 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_FILE.write_text(json.dumps(comparison, indent=2))
    plot_path = plot_comparison(comparison)
    print(
        f"\nNumPy:     test acc {numpy_result['test_acc']:.4f} in {numpy_result['train_time_s']:.1f}s\n"
        f"{framework_name}: test acc {framework_result['test_acc']:.4f} in {framework_result['train_time_s']:.1f}s\n"
        f"Gap: {comparison['test_acc_gap']:.4f}\nComparison: {COMPARISON_FILE}\nPlot: {plot_path}"
    )
    return comparison
