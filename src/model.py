"""
model.py - Convolutional Neural Network definition for EMNIST letter classification.
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN Architecture:
    Input: 1 x 28 x 28
    Conv 3x3, 32 -> ReLU -> Conv 3x3, 64 -> ReLU -> MaxPool 2x2 -> Dropout 0.25
    Flatten -> Linear 128 -> ReLU -> Dropout 0.5 -> Linear 26
    """
    def __init__(self, num_classes=26):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv 1: 1 -> 32 channels, 3x3 kernel, unpadded (28 -> 26)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            # Conv 2: 32 -> 64 channels, 3x3 kernel, unpadded (26 -> 24)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            # MaxPool: 2x2 (24 -> 12)
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )
        
        # Flatten size: 64 channels * 12 * 12 = 9216
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
