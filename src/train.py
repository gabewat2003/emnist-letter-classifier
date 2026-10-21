"""
train.py - Train the EMNIST letters CNN.

Selects the best epoch on a held-out validation split and saves it to
checkpoints/best.pt. The test set is not touched here; see evaluate.py.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data import get_data_loaders
from src.model import CNN, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax_loss.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    ax_acc.plot(epochs, [100 * h["train_acc"] for h in history], marker="o", label="train")
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], marker="o", label="validation")
    ax_acc.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(epochs=10, batch_size=128, lr=1e-3, seed=42, device_str=None, num_workers=2):
    torch.manual_seed(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({device_name})")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size, seed=seed, num_workers=num_workers)
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    model = CNN(num_classes=26).to(device)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history, best_acc, best_epoch = [], 0.0, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
        print(f"Epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} acc {100*train_acc:.2f}% "
              f"| val loss {val_loss:.4f} acc {100*val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()},
                       "checkpoints/best.pt")

    train_time = time.time() - start
    print(f"Done in {train_time:.1f}s. Best val acc {100*best_acc:.2f}% at epoch {best_epoch}.")

    plot_history(history, "results/training_curves.png")
    summary = dict(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, optimizer="Adam",
                   device=device_name, train_time_sec=round(train_time, 1), parameters=n_params,
                   best_epoch=best_epoch, best_val_acc=best_acc, history=history)
    Path("results/training.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EMNIST letters CNN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    train_model(args.epochs, args.batch_size, args.lr, args.seed, args.device, args.num_workers)
