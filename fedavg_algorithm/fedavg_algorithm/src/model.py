import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(dataset: str) -> nn.Module:
    if dataset in ["mnist", "fmnist"]:
        return SimpleCNN(in_channels=1, num_classes=10)
    elif dataset == "cifar10":
        return SimpleCNN(in_channels=3, num_classes=10)
    elif dataset == "cifar100":
        return SimpleCNN(in_channels=3, num_classes=100)
    else:
        raise ValueError(f"No model defined for dataset: {dataset}")


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)