"""
data.py - Dataset loading and preprocessing for EMNIST Letters.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = [chr(ord("A") + i) for i in range(26)]


class Transpose:
    """Raw EMNIST images are stored transposed; swap H and W so letters read upright."""

    def __call__(self, x):
        return x.transpose(1, 2)


def shift_label(y):
    """EMNIST letters labels run 1-26; map them to 0-25."""
    return y - 1


def get_transforms():
    return transforms.Compose([transforms.ToTensor(), Transpose()])


def get_datasets(data_dir="./data", val_fraction=0.1, seed=42):
    """Return (train, val, test) datasets. Validation is a fixed slice of the official train split."""
    kwargs = dict(root=data_dir, split="letters", download=True,
                  transform=get_transforms(), target_transform=shift_label)
    full_train = datasets.EMNIST(train=True, **kwargs)
    test = datasets.EMNIST(train=False, **kwargs)

    n_val = int(len(full_train) * val_fraction)
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [len(full_train) - n_val, n_val], generator=generator)
    return train, val, test


def get_data_loaders(data_dir="./data", batch_size=128, val_fraction=0.1, seed=42, num_workers=2):
    """Return (train_loader, val_loader, test_loader)."""
    train, val, test = get_datasets(data_dir, val_fraction, seed)
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
                  persistent_workers=num_workers > 0)
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )
