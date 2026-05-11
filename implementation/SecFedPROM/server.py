"""
server.py — Flower FL Server with SecureFedPROM Strategy
Implements custom FedAvg strategy that integrates ABAC + PROMETHEE client selection.
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (Parameters, FitIns, FitRes, EvaluateIns,
                          EvaluateRes, Scalar, ndarrays_to_parameters,
                          parameters_to_ndarrays)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from model import get_model, get_model_parameters, set_model_parameters
from utils import (ABACController, PROMETHEESelector, ClientProfile,
                   evaluate_global, fedavg_aggregate,
                   random_selection, power_of_choice_selection,
                   greedy_selection, resource_aware_selection,
                   price_first_selection)


class SecureFedPROMStrategy(FedAvg):
    """
    Custom Flower strategy implementing the full SecureFedPROM protocol.

    Extends FedAvg with:
      1. ABAC authorization filter
      2. PROMETHEE multi-criteria client ranking
      3. Dynamic trust score updates after each round
      4. Budget-constrained selection
    """

    def __init__(self,
                 client_profiles: List[ClientProfile],
                 dataset: str,
                 num_classes: int,
                 selection_strategy: str = "securefedprom",
                 clients_per_round: int = 10,
                 budget: float = 50.0,
                 weights: Optional[Dict[str, float]] = None,
                 test_loader=None,
                 device: str = "cpu",
                 **kwargs):
        super().__init__(**kwargs)

        self.client_profiles      = {c.client_id: c for c in client_profiles}
        self.dataset              = dataset
        self.num_classes          = num_classes
        self.selection_strategy   = selection_strategy.lower()
        self.clients_per_round    = clients_per_round
        self.budget               = budget
        self.test_loader          = test_loader
        self.device               = device

        # Core SecureFedPROM components
        self.abac      = ABACController()
        self.promethee = PROMETHEESelector(weights=weights)

        # Metrics tracking
        self.round_metrics: List[Dict] = []
        self.authorized_ids: List[int] = []

        # Run ABAC once at init
        all_profiles = list(self.client_profiles.values())
        authorized   = self.abac.filter_clients(all_profiles)
        self.authorized_ids = [c.client_id for c in authorized]
        print(f"[Server] Strategy: {selection_strategy.upper()}")
        print(f"[Server] Authorized clients: {len(self.authorized_ids)}")

    def _get_selected_client_ids(self, round_num: int) -> List[int]:
        """
        Select clients for this round based on chosen strategy.
        Only authorized clients are eligible.
        """
        eligible = [self.client_profiles[cid]
                    for cid in self.authorized_ids
                    if cid in self.client_profiles]

        k = self.clients_per_round

        if self.selection_strategy == "securefedprom":
            selected = self.promethee.select_clients(eligible, k, self.budget)

        elif self.selection_strategy == "random":
            selected = random_selection(eligible, k)

        elif self.selection_strategy == "power_of_choice":
            selected = power_of_choice_selection(eligible, k)

        elif self.selection_strategy == "greedy":
            selected = greedy_selection(eligible, k)

        elif self.selection_strategy == "resource_aware":
            selected = resource_aware_selection(eligible, k)

        elif self.selection_strategy == "price_first":
            selected = price_first_selection(eligible, k, self.budget)

        else:
            selected = random_selection(eligible, k)

        return [c.client_id for c in selected]

    def configure_fit(self, server_round: int,
                      parameters: Parameters,
                      client_manager) -> List[Tuple]:
        """Configure which clients participate in this round."""
        selected_ids = self._get_selected_client_ids(server_round)

        config = {
            "round": server_round,
            "local_epochs": 5,
            "lr": 0.01
        }

        # Get all available clients and filter to selected
        all_clients = client_manager.all()
        fit_configs = []
        for cid_str, client_proxy in all_clients.items():
            cid = int(cid_str) if cid_str.isdigit() else hash(cid_str) % len(self.client_profiles)
            if cid in selected_ids:
                fit_configs.append((client_proxy, FitIns(parameters, config)))

        # Fallback: if mapping fails, use fraction sampling
        if not fit_configs:
            sample_size = max(1, int(len(all_clients) * 0.5))
            sampled = client_manager.sample(num_clients=sample_size)
            fit_configs = [(c, FitIns(parameters, config)) for c in sampled]

        return fit_configs

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple],
                      failures: List) -> Tuple[Optional[Parameters], Dict]:
        """
        Aggregate model updates using weighted FedAvg (Eq. 2).
        Update trust scores based on participation success/failure.
        """
        if not results:
            return None, {}

        # Update trust scores for participants
        for _, fit_res in results:
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                cid = fit_res.metrics.get("client_id", -1)
                if cid in self.client_profiles:
                    self.abac.update_trust(self.client_profiles[cid],
                                           success=True)

        # Aggregate using parent FedAvg
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures)

        return aggregated_params, metrics

    def evaluate(self, server_round: int,
                 parameters: Parameters) -> Optional[Tuple[float, Dict]]:
        """Centralized evaluation on global test set."""
        if self.test_loader is None:
            return None

        model = get_model(self.dataset, self.num_classes)
        ndarrays = parameters_to_ndarrays(parameters)
        set_model_parameters(model, ndarrays)

        from torch import nn
        criterion = nn.CrossEntropyLoss()
        accuracy, loss = evaluate_global(model, self.test_loader,
                                          criterion, self.device)

        print(f"  Round {server_round:3d} | "
              f"Accuracy: {accuracy:.2f}% | Loss: {loss:.4f}")

        self.round_metrics.append({
            "round":    server_round,
            "accuracy": accuracy,
            "loss":     loss
        })

        return loss, {"accuracy": accuracy, "round": server_round}
