"""
client.py — Flower FL Client for SecureFedPROM experiments
Each FlowerClient wraps a local model and dataloader.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple
import copy

from model import get_model, get_model_parameters, set_model_parameters
from utils import train_local, evaluate_global


class FlowerClient(fl.client.NumPyClient):
    """
    Standard Flower client implementing local training (FedAvg style).
    Used for ALL strategies — selection happens server-side.
    """

    def __init__(self,
                 client_id: int,
                 dataset: str,
                 train_loader,
                 test_loader,
                 num_classes: int = 10,
                 local_epochs: int = 5,
                 lr: float = 0.01,
                 device: str = "cpu"):
        self.client_id    = client_id
        self.dataset      = dataset
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.num_classes  = num_classes
        self.local_epochs = local_epochs
        self.lr           = lr
        self.device       = device

        self.model     = get_model(dataset, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        set_model_parameters(self.model, parameters)

    def fit(self, parameters: List[np.ndarray],
            config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training — called by server each round."""
        self.set_parameters(parameters)

        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=1e-4)

        avg_loss, train_time = train_local(
            self.model, self.train_loader,
            optimizer, self.criterion,
            epochs=self.local_epochs,
            device=self.device
        )

        num_samples = len(self.train_loader.dataset)
        return (get_model_parameters(self.model),
                num_samples,
                {"loss": avg_loss, "train_time": train_time,
                 "client_id": self.client_id})

    def evaluate(self, parameters: List[np.ndarray],
                 config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate local model on test data."""
        self.set_parameters(parameters)
        accuracy, loss = evaluate_global(
            self.model, self.test_loader,
            self.criterion, self.device
        )
        num_samples = len(self.test_loader.dataset)
        return loss, num_samples, {"accuracy": accuracy}
