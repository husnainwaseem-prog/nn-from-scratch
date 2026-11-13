"""Command-line interface.

Usage (from the project root, with the venv active):
    python -m src.cli train            Train the NumPy network, save history + curves
    python -m src.cli gradient-check   Verify backprop against numerical gradients
    python -m src.cli compare          Train NumPy vs PyTorch (or sklearn MLP fallback)
    python -m src.cli evaluate         Reload the training history and print final metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import run_comparison
from .data import describe, load_data
from .gradient_check import gradient_check
from .train import HISTORY_FILE, run_training


def _cmd_train(args: argparse.Namespace) -> None:
    X_train, y_train, X_test, y_test = load_data()
    print("Data:", describe(X_train, y_train, X_test, y_test))
    run_training(epochs=args.epochs, lr=args.lr, momentum=args.momentum, batch_size=args.batch_size)


def _cmd_gradient_check(args: argparse.Namespace) -> None:
    report = gradient_check()
    if not report["passed"]:
        raise SystemExit("Gradient check FAILED — backprop does not match numerical gradients.")


def _cmd_compare(args: argparse.Namespace) -> None:
    run_comparison()


def _cmd_evaluate(args: argparse.Namespace) -> None:
    history_path = Path(HISTORY_FILE)
    if not history_path.exists():
        print("No training history yet — run `python -m src.cli train` first.")
        raise SystemExit(1)
    history = json.loads(history_path.read_text())
    print(f"Final train accuracy: {history['final_train_acc']:.4f}")
    print(f"Final test accuracy:  {history['final_test_acc']:.4f}")
    print(f"Training time:        {history['train_time_s']}s over {len(history['train_loss'])} epochs")
    print("Per-epoch test accuracy:", history["test_acc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural network from scratch (NumPy).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the NumPy network on MNIST.")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=0.05)
    p_train.add_argument("--momentum", type=float, default=0.9)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.set_defaults(func=_cmd_train)

    p_gc = sub.add_parser("gradient-check", help="Verify backprop with numerical gradients.")
    p_gc.set_defaults(func=_cmd_gradient_check)

    p_cmp = sub.add_parser("compare", help="Compare NumPy vs PyTorch (or sklearn MLP fallback).")
    p_cmp.set_defaults(func=_cmd_compare)

    p_eval = sub.add_parser("evaluate", help="Print final metrics from the last training run.")
    p_eval.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
