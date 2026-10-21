"""
evaluate.py - Test-set metrics, confusion matrix, and error gallery.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.data import CLASSES, get_data_loaders
from src.model import CNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Train first: python -m src.train")
    model = CNN(num_classes=26).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


@torch.no_grad()
def predict(model, loader, device):
    preds, targets, images = [], [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        targets.append(y)
        images.append(x)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), torch.cat(images).squeeze(1).numpy()


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(26)
    ax.set_xticks(ticks, CLASSES)
    ax.set_yticks(ticks, CLASSES)
    ax.set(title="EMNIST Letters - Test Confusion Matrix", xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_gallery(pairs, preds, targets, images, path, per_pair=6):
    fig, axes = plt.subplots(len(pairs), per_pair, figsize=(per_pair * 1.4, len(pairs) * 1.7))
    for row, (true_cls, pred_cls, count) in enumerate(pairs):
        idx = np.where((targets == true_cls) & (preds == pred_cls))[0][:per_pair]
        for col in range(per_pair):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(idx):
                ax.imshow(images[idx[col]], cmap="gray")
        axes[row, 0].set_title(f"true {CLASSES[true_cls]} -> pred {CLASSES[pred_cls]} ({count})",
                               loc="left", fontsize=9)
    fig.suptitle("Most-confused pairs (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_model(checkpoint_path="checkpoints/best.pt", num_workers=2):
    device = get_device()
    print(f"Device: {device}")
    _, _, test_loader = get_data_loaders(num_workers=num_workers)
    model = load_model(checkpoint_path, device)

    preds, targets, images = predict(model, test_loader, device)
    accuracy = float((preds == targets).mean())
    print(f"Test accuracy: {100 * accuracy:.2f}% on {len(targets):,} images")

    cm = confusion_matrix(targets, preds, labels=range(26))
    per_class = cm.diagonal() / cm.sum(axis=1)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    flat = np.argsort(off_diag, axis=None)[::-1][:3]
    pairs = [(int(i), int(j), int(off_diag[i, j])) for i, j in zip(*np.unravel_index(flat, cm.shape))]

    print("Most-confused pairs (true -> predicted):")
    for t, p, c in pairs:
        print(f"  {CLASSES[t]} -> {CLASSES[p]}: {c}")

    Path("results").mkdir(exist_ok=True)
    plot_confusion_matrix(cm, "results/confusion_matrix.png")
    plot_error_gallery(pairs, preds, targets, images, "results/error_gallery.png")
    metrics = dict(
        test_accuracy=accuracy,
        test_images=int(len(targets)),
        per_class_accuracy={CLASSES[i]: float(a) for i, a in enumerate(per_class)},
        most_confused=[dict(true=CLASSES[t], predicted=CLASSES[p], count=c) for t, p, c in pairs],
    )
    Path("results/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved results/confusion_matrix.png, results/error_gallery.png, results/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the EMNIST letters CNN on the test set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.num_workers)
