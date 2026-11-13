"""Data loading: MNIST, cached locally so the download happens once.

MNIST is 70,000 greyscale images of handwritten digits (28x28 pixels),
the standard first dataset for a from-scratch network: 60,000 train /
10,000 test, 10 classes. Pixels are scaled from 0-255 to 0-1, which
keeps the inputs in the range where the activations and gradients
behave well.

The download comes from OpenML (sklearn's fetch_openml). The raw
arrays are cached as data/mnist.npz so later runs — and the laptop
recording — never need the network again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "mnist.npz"


def _download_mnist_openml() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST with sklearn's fetch_openml (the standard route)."""
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(np.int64)
    return X[:60_000], y[:60_000], X[60_000:], y[60_000:]


def _download_mnist_arff() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download the same MNIST release as an ARFF file from OpenML and
    parse it with scipy. Used when fetch_openml's parser fails (it
    needs a matching pandas version, which is not guaranteed)."""
    import gzip
    import io
    import urllib.request

    from scipy.io import arff

    url = "https://openml.org/data/v1/download/52667"
    with urllib.request.urlopen(url, timeout=180) as resp:
        raw = resp.read()
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except Exception:
        # The endpoint sometimes serves the ARFF uncompressed.
        text = raw.decode("utf-8")
    data, meta = arff.loadarff(io.StringIO(text))
    pixel_names = meta.names()[:-1]  # 784 pixel columns; last column is 'class'
    X = np.column_stack([data[name] for name in pixel_names]).astype(np.float64) / 255.0
    y = np.array([int(c) for c in data["class"]], dtype=np.int64)
    return X[:60_000], y[:60_000], X[60_000:], y[60_000:]


def _download_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST from OpenML and cache it as data/mnist.npz."""
    print("Downloading MNIST from OpenML (this happens once, ~15 MB)...")
    try:
        X_train, y_train, X_test, y_test = _download_mnist_openml()
    except Exception as exc:  # noqa: BLE001 — try the ARFF route instead
        print(f"fetch_openml failed ({exc}); trying the ARFF download instead...")
        X_train, y_train, X_test, y_test = _download_mnist_arff()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Cached MNIST to {CACHE_FILE}")
    return X_train, y_train, X_test, y_test


def _load_digits_fallback() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback if MNIST cannot be downloaded: sklearn's built-in
    8x8 digits dataset (1,797 images). Same shape of problem, smaller
    scale — the network code is unchanged, only the input size (64)
    and the split differ."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    print("WARNING: MNIST download failed — falling back to sklearn digits (8x8).")
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test).

    Tries the local cache first, then the OpenML download (with one
    retry), then the digits fallback. Never raises for a missing
    network connection — the fallback keeps the project runnable.
    """
    if CACHE_FILE.exists():
        cached = np.load(CACHE_FILE)
        return cached["X_train"], cached["y_train"], cached["X_test"], cached["y_test"]
    try:
        return _download_mnist()
    except Exception as exc:  # noqa: BLE001 — any download failure falls back
        print(f"MNIST download failed: {exc}")
    return _load_digits_fallback()


def describe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> str:
    """One-line summary of the loaded split, for logs and the report."""
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    return (
        f"train {X_train.shape[0]} x {X_train.shape[1]} | "
        f"test {X_test.shape[0]} x {X_test.shape[1]} | "
        f"{n_classes} classes | pixel range [{X_train.min():.2f}, {X_train.max():.2f}]"
    )
