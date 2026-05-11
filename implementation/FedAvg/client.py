import random
import time
import torch
import torch.nn as nn
import flwr as fl
from data import get_client_dataloader, set_seeds


class FLClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid,
        model,
        train_indices,
        train_dataset,
        device,
        local_epochs,
        batch_size,
        lr,
        seed,
        straggler_prob=0.0,
        straggler_delay=(0, 0),
    ):
        self.cid             = cid
        self.model           = model.to(device)
        self.train_indices   = train_indices
        self.train_dataset   = train_dataset
        self.device          = device
        self.local_epochs    = local_epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.seed            = seed
        self.straggler_prob  = straggler_prob
        self.straggler_delay = straggler_delay

    # ------------------------------------------------------------------ #

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if len(self.train_indices) == 0:
            print(f"  [Client {self.cid}] WARNING: empty partition — skipping fit, "
                  f"returning current parameters unchanged.")
            return (
                self.get_parameters({}),
                0,                          # 0 samples → server weights this client at 0
                {
                    "train_loss":     0.0,
                    "train_accuracy": 0.0,
                    "straggler":      0,
                },
            )

        set_seeds(self.seed)
        self.set_parameters(parameters)
        self.model.train()

        # ---- straggler simulation ----
        is_straggler = 0
        if self.straggler_prob > 0 and random.random() < self.straggler_prob:
            delay = random.randint(self.straggler_delay[0], self.straggler_delay[1])
            time.sleep(delay)
            is_straggler = 1

        loader    = get_client_dataloader(self.train_dataset, self.train_indices, self.batch_size)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_fn   = nn.CrossEntropyLoss()

        total_loss, correct, total = 0.0, 0, 0

        for _ in range(self.local_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X)
                loss   = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y)
                correct    += (logits.argmax(1) == y).sum().item()
                total      += len(y)

        avg_loss = total_loss / max(total, 1)
        avg_acc  = correct    / max(total, 1)

        return (
            self.get_parameters({}),
            len(self.train_indices),
            {
                "train_loss":     float(avg_loss),
                "train_accuracy": float(avg_acc),
                "straggler":      int(is_straggler),
            },
        )

    # ------------------------------------------------------------------ #

    def evaluate(self, parameters, config):
        # ---- FIX: guard against empty partition ----
        if len(self.train_indices) == 0:
            print(f"  [Client {self.cid}] WARNING: empty partition — skipping evaluate.")
            return (
                0.0,
                0,
                {"accuracy": 0.0},
            )

        self.set_parameters(parameters)
        self.model.eval()
        loader  = get_client_dataloader(
            self.train_dataset, self.train_indices, self.batch_size, shuffle=False
        )
        loss_fn = nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y    = X.to(self.device), y.to(self.device)
                logits  = self.model(X)
                loss   += loss_fn(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total   += len(y)
        return (
            loss / max(total, 1),
            len(self.train_indices),
            {"accuracy": correct / max(total, 1)},
        )
