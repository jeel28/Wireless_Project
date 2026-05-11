import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(dataset: str):
    if dataset in ["mnist", "fmnist"]:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset in ["cifar10", "cifar100"]:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def load_dataset(dataset: str, data_dir: str = "./data"):
    tf = get_transforms(dataset)
    ds_map = {
        "mnist":    (datasets.MNIST,    datasets.MNIST),
        "fmnist":   (datasets.FashionMNIST, datasets.FashionMNIST),
        "cifar10":  (datasets.CIFAR10,  datasets.CIFAR10),
        "cifar100": (datasets.CIFAR100, datasets.CIFAR100),
    }
    TrainDS, TestDS = ds_map[dataset]
    train_ds = TrainDS(data_dir, train=True,  download=True, transform=tf)
    test_ds  = TestDS(data_dir,  train=False, download=True, transform=tf)
    return train_ds, test_ds


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42
) -> List[List[int]]:
    """
    Partition dataset indices among clients using Dirichlet distribution.
    alpha=IID means uniform split. alpha -> 0 means extreme non-IID.
    """
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)
        if alpha == float("inf"):  # IID
            splits = np.array_split(cls_idx, num_clients)
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())
        else:
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.cumsum(proportions)
            split_points = (proportions[:-1] * len(cls_idx)).astype(int)
            splits = np.split(cls_idx, split_points)
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())

    return client_indices


def get_client_dataloader(
    dataset,
    indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_test_dataloader(dataset, batch_size: int = 64) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)