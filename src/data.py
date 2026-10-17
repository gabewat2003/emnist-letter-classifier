"""
data.py - Dataset loading and preprocessing transforms for EMNIST Letters.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class EMNISTTransposeTransform:
    """
    Transform to correct the default EMNIST rotation/transposition.
    Raw EMNIST images are typically rotated 90 degrees clockwise and flipped,
    so we transpose and flip them to display properly.
    """
    def __call__(self, x):
        # Transpose and flip horizontally/vertically as needed for correct orientation
        return x.permute(0, 2, 1)

def get_transforms():
    """Returns the composition of transforms for EMNIST."""
    return transforms.Compose([
        transforms.ToTensor(),
        EMNISTTransposeTransform(),
    ])

def get_data_loaders(data_dir="./data", batch_size=128, train_val_split=0.9):
    """
    Downloads and returns train, validation, and test data loaders for EMNIST letters.
    EMNIST letters labels range from 1 to 26; we handle label adjustment during training/evaluation
    or via a custom wrapper if needed. Here we return standard datasets and handle labels in training.
    """
    transform = get_transforms()
    
    train_dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, train_dataset, test_dataset
