"""
data.py — Data loading and Non-IID partitioning using Dirichlet distribution
As required by the professor's guidelines: alpha in {0.01, 0.1, 0.5, 1.0, IID}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
import random


# ─── Fixed seeds as required by professor ────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_transforms(dataset: str):
    """Returns train and test transforms for each dataset."""
    dataset = dataset.lower()
    if dataset in ["fmnist", "femnist", "mnist"]:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset in ["cifar10", "cifar-10"]:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_tf, test_tf


def load_dataset(dataset: str, data_root: str = "./data"):
    """Load the full train and test datasets."""
    dataset = dataset.lower()
    train_tf, test_tf = get_transforms(dataset)

    if dataset in ["fmnist", "femnist"]:
        train_ds = datasets.FashionMNIST(data_root, train=True,
                                          download=True, transform=train_tf)
        test_ds  = datasets.FashionMNIST(data_root, train=False,
                                          download=True, transform=test_tf)
    elif dataset in ["cifar10", "cifar-10"]:
        train_ds = datasets.CIFAR10(data_root, train=True,
                                     download=True, transform=train_tf)
        test_ds  = datasets.CIFAR10(data_root, train=False,
                                     download=True, transform=test_tf)
    elif dataset == "mnist":
        train_ds = datasets.MNIST(data_root, train=True,
                                   download=True, transform=train_tf)
        test_ds  = datasets.MNIST(data_root, train=False,
                                   download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_ds, test_ds


def dirichlet_partition(dataset, num_clients: int,
                        alpha: float,
                        num_classes: int = 10) -> List[List[int]]:
    """
    Partition dataset indices among clients using Dirichlet distribution.

    alpha controls heterogeneity:
        alpha → 0    : extreme Non-IID (each client gets 1 class)
        alpha = 0.1  : high Non-IID
        alpha = 0.5  : moderate Non-IID
        alpha = 1.0  : mild Non-IID
        alpha → ∞    : IID

    Returns: List of index lists, one per client.
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        # Draw proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Normalize so they sum to the class count
        proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, proportions)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # Shuffle each client's indices
    for i in range(num_clients):
        random.shuffle(client_indices[i])

    return client_indices


def iid_partition(dataset, num_clients: int) -> List[List[int]]:
    """Partition dataset indices uniformly (IID) among clients."""
    n = len(dataset)
    all_idx = list(range(n))
    random.shuffle(all_idx)
    size = n // num_clients
    return [all_idx[i * size:(i + 1) * size] for i in range(num_clients)]


def get_client_dataloader(dataset, indices: List[int],
                          batch_size: int = 32,
                          shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for a client's subset of data."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def get_test_dataloader(test_dataset, batch_size: int = 64) -> DataLoader:
    """Global test DataLoader."""
    return DataLoader(test_dataset, batch_size=batch_size,
                      shuffle=False, num_workers=0)


def partition_data(dataset_name: str,
                   num_clients: int,
                   alpha: float,
                   batch_size: int = 32,
                   data_root: str = "./data",
                   num_classes: int = 10):
    """
    Full pipeline: load data → partition → return per-client loaders + test loader.

    Args:
        dataset_name : 'fmnist' or 'cifar10'
        num_clients  : number of FL clients (10, 50, 100)
        alpha        : Dirichlet alpha (0.01, 0.1, 0.5, 1.0) or float('inf') for IID
        batch_size   : local batch size (32 as per professor)

    Returns:
        client_loaders : list of DataLoaders, one per client
        test_loader    : global test DataLoader
        num_classes    : number of output classes
    """
    train_ds, test_ds = load_dataset(dataset_name, data_root)

    if alpha == float('inf') or alpha <= 0:
        print(f"[Data] IID partitioning — {num_clients} clients")
        client_idx = iid_partition(train_ds, num_clients)
    else:
        print(f"[Data] Non-IID Dirichlet(α={alpha}) — {num_clients} clients")
        client_idx = dirichlet_partition(train_ds, num_clients,
                                          alpha, num_classes)

    client_loaders = [
        get_client_dataloader(train_ds, idx, batch_size)
        for idx in client_idx
    ]
    test_loader = get_test_dataloader(test_ds)

    sizes = [len(idx) for idx in client_idx]
    print(f"[Data] Samples per client — min:{min(sizes)}, "
          f"max:{max(sizes)}, mean:{np.mean(sizes):.1f}")

    return client_loaders, test_loader, num_classes
