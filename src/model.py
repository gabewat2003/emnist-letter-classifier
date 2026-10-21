"""
model.py - CNN for EMNIST letter classification.
"""

import torch.nn as nn


class CNN(nn.Module):
    """
    Input 1x28x28
    Conv 3x3,32 -> ReLU -> Conv 3x3,64 -> ReLU -> MaxPool 2x2 -> Dropout 0.25
    Flatten -> Linear 128 -> ReLU -> Dropout 0.5 -> Linear 26
    """

    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 28 -> 26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),  # 26 -> 24
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 24 -> 12
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
