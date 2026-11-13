"""Tests for the from-scratch neural network.

Run: python -m pytest tests/ -q
"""

import numpy as np
import pytest

from src.activations import relu, sigmoid, softmax, tanh
from src.gradient_check import gradient_check
from src.network import NeuralNetwork, one_hot
from src.optimizer import SGDMomentum
from src.train import train


def test_forward_output_shape():
    net = NeuralNetwork([784, 128, 64, 10])
    out = net.forward(np.random.default_rng(0).normal(size=(32, 784)))
    assert out.shape == (32, 10)


def test_softmax_rows_sum_to_one():
    x = np.random.default_rng(1).normal(size=(50, 10)) * 5
    probs = softmax(x)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert (probs >= 0).all()


def test_activations_known_values():
    assert np.allclose(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])
    assert np.isclose(sigmoid(0.0), 0.5)
    assert np.isclose(tanh(0.0), 0.0)
    assert np.allclose(tanh(np.array([0.0])), [0.0])


def test_gradient_check_passes():
    report = gradient_check(verbose=False)
    assert report["passed"], f"max relative error {report['max_relative_error']:.2e} >= 1e-6"
    assert report["max_relative_error"] < 1e-6


def test_loss_decreases_over_short_run():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20))
    y = rng.integers(0, 3, size=500)
    net = NeuralNetwork([20, 16, 3], seed=0)
    loss_before = net.loss(X, y)
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    for _ in range(30):
        idx = rng.integers(0, 500, size=64)
        net.forward(X[idx])
        grads = net.backward(X[idx], one_hot(y[idx], 3))
        opt.update(net.params, grads)
    loss_after = net.loss(X, y)
    assert loss_after < loss_before


def test_momentum_optimizer_updates_velocity():
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    params = {"W1": np.zeros((2, 2))}
    grads = {"W1": np.ones((2, 2))}
    opt.update(params, grads)
    # First step: velocity = -lr * grad = -0.1
    assert np.allclose(params["W1"], -0.1)
    opt.update(params, grads)
    # Second step: velocity = 0.9 * (-0.1) - 0.1 = -0.19, so W = -0.1 - 0.19
    assert np.allclose(params["W1"], -0.29)


def test_mini_batch_shapes():
    net = NeuralNetwork([10, 8, 4])
    rng = np.random.default_rng(2)
    X = rng.normal(size=(7, 10))
    y = rng.integers(0, 4, size=7)
    net.forward(X)
    grads = net.backward(X, one_hot(y, 4))
    for key, param in net.params.items():
        assert grads[key].shape == param.shape, f"{key}: {grads[key].shape} != {param.shape}"


def test_predict_returns_valid_classes():
    net = NeuralNetwork([10, 8, 4], seed=3)
    preds = net.predict(np.random.default_rng(3).normal(size=(20, 10)))
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3})


def test_training_reaches_accuracy_on_digits():
    """End-to-end: 5 epochs on the small digits dataset must clear 0.90.
    (The full MNIST run is checked separately in REPORT.md; this keeps
    the suite fast.)"""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    _, history = train(
        layer_sizes=[64, 32, 10],
        epochs=5,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        verbose=False,
    )
    assert history["final_test_acc"] >= 0.90, f"digits test acc {history['final_test_acc']:.4f}"


def test_one_hot_encoding():
    oh = one_hot(np.array([0, 2, 1]), 3)
    assert oh.shape == (3, 3)
    assert np.allclose(oh, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_mnist_final_accuracy_at_least_095():
    """The real MNIST run (python -m src.cli train) must clear 0.95 test
    accuracy. Reads the saved history; skipped until training has run."""
    import json
    from pathlib import Path

    history_file = Path(__file__).resolve().parent.parent / "results" / "training_history.json"
    if not history_file.exists():
        pytest.skip("no training run yet — run `python -m src.cli train` first")
    history = json.loads(history_file.read_text())
    assert history["final_test_acc"] >= 0.95, f"MNIST test acc {history['final_test_acc']:.4f}"
