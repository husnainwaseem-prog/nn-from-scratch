# Neural Network from Scratch

A feedforward neural network built **only with NumPy** — no PyTorch, no
TensorFlow, no Keras for the network itself. It classifies handwritten
digits (MNIST, 60,000 train / 10,000 test) and is compared against the
same architecture built in PyTorch.

## What each part does, and why

| File | What it does | Why it exists |
|---|---|---|
| `src/activations.py` | ReLU, sigmoid, tanh (+ derivatives) and softmax | A stack of linear layers is still just linear. The activation is the non-linearity that lets the network learn curved boundaries; its derivative is what backpropagation needs. Softmax turns the output into a probability distribution over the 10 digits. |
| `src/network.py` | The network: forward pass, backpropagation, cross-entropy loss | The core. Forward pushes a batch through the layers, caching every intermediate value. Backward uses the chain rule to work out each weight's contribution to the loss — for softmax + cross-entropy the output error is simply `(probabilities − labels) / batch_size`. Weights use He initialisation for ReLU layers so the signal neither shrinks nor explodes. |
| `src/optimizer.py` | SGD with momentum | Plain gradient descent zig-zags in narrow valleys. Momentum keeps a running average of past gradients, so consistent directions speed up and opposing ones cancel — like a heavy ball rolling downhill. |
| `src/data.py` | Loads MNIST, caches it as `data/mnist.npz` | Pixels are scaled 0–255 → 0–1 so inputs sit in the range where activations and gradients behave. The download is tried with `fetch_openml` first, then as a raw ARFF parsed with scipy (the pandas-based parser fails on some setups), then falls back to sklearn's small digits set so the project still runs offline. |
| `src/train.py` | Mini-batch training loop + curves | One epoch = one shuffled pass over the data in batches of 128. After each epoch it measures loss/accuracy on train and test — the curves show whether the network is learning (loss falling) and generalising (test tracking train). History is saved to `results/training_history.json`. |
| `src/gradient_check.py` | Numerical gradient checking | Backprop is easy to get subtly wrong and the network will still "train". This nudges each weight by ±ε and compares the loss change against backprop's gradient. Max relative error must be < 1e-6 (measured: 6.63e-08). Runs on a tiny 4→5→3 network because the numerical pass needs two forward passes per weight. |
| `src/compare.py` | NumPy vs PyTorch, same everything | Same architecture, optimizer, learning rate, batch size, epochs, split and seed — so any gap is a real difference in implementation, not setup. If PyTorch is missing it falls back to sklearn's MLPClassifier and says so in the report. |
| `src/cli.py` | Command-line interface | Four commands: `train`, `gradient-check`, `compare`, `evaluate`. |

## Architecture and training

- **784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)**, cross-entropy loss
- **SGD, momentum 0.9, learning rate 0.05, batch size 128, 10 epochs, seed 42**
- 784 inputs = one 28×28 MNIST image flattened; 10 outputs = one per digit

## Results

See [REPORT.md](REPORT.md) for the measured numbers: per-epoch loss and
accuracy, the gradient-check error, and the NumPy-vs-PyTorch comparison
(accuracy and training time). Training curves are in
`results/plots/training_curves.png`, the comparison in
`results/plots/comparison.png`.

## Setup and usage

```bash
python -m venv .venv
# Windows PowerShell:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m src.cli gradient-check   # verify backprop (should PASS)
python -m src.cli train            # train 10 epochs, save curves
python -m src.cli compare          # NumPy vs PyTorch
python -m src.cli evaluate         # print final metrics
python -m pytest tests/ -q         # test suite
```

The first run downloads MNIST from OpenML (~15 MB) and caches it in
`data/mnist.npz`; later runs work offline.

## Tests

11 tests (`tests/test_nn.py`): forward shapes, softmax rows summing to 1,
activation values, gradient check passing, loss decreasing over a short
run, momentum velocity updates, mini-batch gradient shapes, valid
predictions, one-hot encoding, an end-to-end run on the digits set, and
the real MNIST run clearing 0.95 test accuracy.

## Docker

```bash
docker build -t nn-from-scratch .
docker run --rm nn-from-scratch
```

The container runs the same `python -m src.cli train` command and
downloads MNIST on first run.
