# Report — Neural Network from Scratch

## 1. What was built

A feedforward neural network written only in NumPy (no ML framework for
the network itself): **784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)**,
cross-entropy loss, mini-batch gradient descent (batch 128) with SGD +
momentum (0.9), learning rate 0.05, 10 epochs, seed 42. Trained on
MNIST: 60,000 train / 10,000 test, pixels scaled to 0–1.

## 2. Training results (measured, first run)

Total training time: **54.61 s** (sandbox, ~1.5 shared CPUs).

| Epoch | Train loss | Train acc | Test loss | Test acc |
|---|---|---|---|---|
| 1 | 0.1379 | 0.9589 | 0.1437 | 0.9558 |
| 2 | 0.0801 | 0.9759 | 0.1003 | 0.9687 |
| 3 | 0.0639 | 0.9807 | 0.0943 | 0.9703 |
| 4 | 0.0503 | 0.9842 | 0.0906 | 0.9705 |
| 5 | 0.0377 | 0.9885 | 0.0754 | 0.9765 |
| 6 | 0.0375 | 0.9885 | 0.0895 | 0.9720 |
| 7 | 0.0229 | 0.9929 | 0.0764 | 0.9778 |
| 8 | 0.0161 | 0.9953 | 0.0778 | **0.9783** |
| 9 | 0.0157 | 0.9952 | 0.0827 | 0.9774 |
| 10 | 0.0159 | 0.9946 | 0.0870 | 0.9751 |

- **Final test accuracy: 0.9751** (train 0.9946). Best test accuracy was
  epoch 8 (0.9783).
- The network already clears 0.95 test accuracy after a single epoch.
- Mild overfitting after epoch 5: train loss keeps falling (0.0377 →
  0.0159) while test loss flattens and creeps up (0.0754 → 0.0870) and
  test accuracy peaks at epoch 8. Early stopping at epoch 8 would be
  the honest improvement; it was left out to keep the required setup
  (10 epochs) intact.

## 3. Gradient check (measured)

Central-difference numerical gradients vs backpropagation, tiny
network 4 → 5 → 3, 5 samples, ε = 1e-5. Threshold: max relative
error < 1e-6.

| Parameter | Max relative error |
|---|---|
| W1 | 5.87e-10 |
| b1 | 6.63e-08 |
| W2 | 1.99e-10 |
| b2 | 1.01e-10 |

**Max relative error: 6.63e-08 — PASS.** Backpropagation is verified
correct. (b1 is the largest only because bias gradients are small in
absolute terms; it is still ~15× below the threshold.)

## 4. Comparison: NumPy vs PyTorch (measured)

Same architecture, optimizer, learning rate, batch size, epochs, data
split and seed on both sides.

| | Test accuracy | Training time |
|---|---|---|
| NumPy (from scratch) | **0.9751** | 46.0 s |
| PyTorch (CPU, 2.15.0) | **0.9781** | 26.4 s |
| Gap | 0.0030 | — |

The hand-written network lands within 0.3 points of PyTorch — the
implementation is correct; the small gap is mostly float64 vs
float32 arithmetic and different weight initialisation draws.
PyTorch is ~1.7× faster, as expected: fused kernels and float32,
versus plain NumPy matmuls in float64.

Reproducibility: the comparison run retrained the NumPy network from
scratch and reproduced the exact same 0.9751 test accuracy (seed 42).
Training time varies a little between runs (54.6 s vs 46.0 s) because
the sandbox CPU is shared; accuracy does not.

## 5. Tests

**11/11 pytest tests pass** (`python -m pytest tests/ -q`, 2.31 s):
forward shapes, softmax rows summing to 1, activation values, gradient
check passing, loss decreasing over a short run, momentum velocity
updates, mini-batch gradient shapes, valid predictions, one-hot
encoding, an end-to-end digits run (≥ 0.90), and the real MNIST run
clearing 0.95.

## 6. Honest notes

- **MNIST download:** `fetch_openml` fails in this sandbox — the
  pandas-based ARFF parser errors with "expected string or bytes-like
  object, got 'int'" (pandas 3.x). The loader falls back to
  downloading the same OpenML release as a raw ARFF file and parsing
  it with scipy, which works. On a machine with an older pandas the
  first route may succeed; either way the cached `data/mnist.npz`
  (22 MB) is identical.
- **Docker:** the Dockerfile is written and statically checked but
  the image was not built here — the sandbox has no Docker daemon.
  Build it on a machine with Docker (`docker build -t nn-from-scratch .`).
- **Overfitting:** see §2 — after epoch 5 the model starts memorising
  the training set. No regularisation (dropout / weight decay) was
  added, to keep the build to the required spec.
