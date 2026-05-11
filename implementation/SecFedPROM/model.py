"""
model.py — Neural network architectures for SecureFedPROM experiments
Supports: FMNIST, CIFAR-10 (as used in the paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMNISTModel(nn.Module):
    """
    2-layer CNN for FEMNIST/FMNIST as described in the SecureFedPROM paper.
    Two conv layers + pooling + dense layer with 2048 units.
    """
    def __init__(self, num_classes: int = 10):
        super(FMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 2048)
        self.fc2   = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFAR10Model(nn.Module):
    """
    Modified ResNet-style CNN for CIFAR-10 as described in the paper.
    Lighter version suitable for FL simulation.
    """
    def __init__(self, num_classes: int = 10):
        super(CIFAR10Model, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(dataset: str, num_classes: int = 10) -> nn.Module:
    """Factory function to get the right model for a dataset."""
    dataset = dataset.lower()
    if dataset in ["fmnist", "femnist", "mnist"]:
        return FMNISTModel(num_classes)
    elif dataset in ["cifar10", "cifar-10"]:
        return CIFAR10Model(num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose fmnist or cifar10.")


def get_model_parameters(model: nn.Module):
    """Extract model parameters as a list of numpy arrays (for Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters):
    """Set model parameters from a list of numpy arrays (for Flower)."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict  = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model
