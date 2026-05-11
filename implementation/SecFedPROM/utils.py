"""
utils.py — Updated Fair-SecureFedPROM Components:
  1. ABAC (Attribute-Based Access Control)
  2. ClientProfile — device attribute vector
  3. PROMETHEE — multi-criteria ranking
  4. Fair-Selection — 90/10 Exploration logic added
  5. Training and evaluation helpers
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CLIENT PROFILE  (attribute vector V_i from the paper)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClientProfile:
    client_id: int
    ram_gb:          float = 4.0
    cpu_cores:       int   = 2
    storage_gb:      float = 10.0
    battery_pct:     float = 1.0
    bandwidth_mbps:  float = 10.0
    latency_ms:      float = 50.0
    data_size:       int   = 500
    local_loss:      float = 1.0
    trust_score:     float = 1.0
    g_H: float = 0.0
    g_N: float = 0.0
    g_Q: float = 0.0
    g_T: float = 0.0
    cost: float = 1.0

    def compute_scores(self,
                       min_ram: float = 1.0, max_ram: float = 64.0,
                       min_bw:  float = 1.0, max_bw:  float = 1000.0,
                       min_lat: float = 1.0, max_lat: float = 200.0,
                       max_ds:  int   = 5000):
        # Hardware score
        ram_util     = min((self.ram_gb - min_ram)  / (max_ram - min_ram), 1.0)
        cpu_util     = min(self.cpu_cores / 8.0, 1.0)
        storage_util = min(self.storage_gb / 64.0,  1.0)
        battery_util = self.battery_pct
        self.g_H = np.mean([ram_util, cpu_util, storage_util, battery_util])

        # Network score
        bw_util  = min((self.bandwidth_mbps - min_bw) / (max_bw - min_bw), 1.0)
        lat_util = 1.0 - min((self.latency_ms - min_lat) / (max_lat - min_lat), 1.0)
        self.g_N = np.mean([bw_util, lat_util])

        # Data quality score
        size_util = min(self.data_size / max_ds, 1.0)
        loss_util = max(0.0, 1.0 - self.local_loss / 3.0)
        self.g_Q  = np.mean([size_util, loss_util])

        # Trust score
        self.g_T = np.clip(self.trust_score, 0.0, 1.0)

        # Cost model
        self.cost = max(0.1, (1.0 - self.g_H) * 2.0 + (1.0 / max(self.bandwidth_mbps, 1)))
        return self

def simulate_clients(num_clients: int, mislabel_fraction: float = 0.05, seed: int = SEED) -> List[ClientProfile]:
    rng = np.random.default_rng(seed)
    clients = []
    hw_tiers = [
        dict(ram_gb=1.5,  cpu_cores=1, storage_gb=2,   battery_pct=0.6),
        dict(ram_gb=3.0,  cpu_cores=2, storage_gb=5,   battery_pct=0.75),
        dict(ram_gb=6.0,  cpu_cores=4, storage_gb=20,  battery_pct=0.9),
        dict(ram_gb=32.0, cpu_cores=8, storage_gb=50,  battery_pct=1.0),
    ]
    net_tiers = [
        dict(bandwidth_mbps=2,   latency_ms=80),
        dict(bandwidth_mbps=20,  latency_ms=30),
        dict(bandwidth_mbps=100, latency_ms=10),
        dict(bandwidth_mbps=500, latency_ms=3),
    ]

    mislabeled_ids = set(rng.choice(num_clients, size=max(1, int(num_clients * mislabel_fraction)), replace=False).tolist())

    for i in range(num_clients):
        hw  = hw_tiers[rng.integers(0, 4)]
        net = net_tiers[rng.integers(0, 4)]
        trust = 0.3 + rng.random() * 0.4 if i in mislabeled_ids else 0.7 + rng.random() * 0.3
        dloss = rng.uniform(0.5, 2.5) if i in mislabeled_ids else rng.uniform(0.2, 1.2)

        profile = ClientProfile(
            client_id=i, ram_gb=hw['ram_gb'], cpu_cores=hw['cpu_cores'], storage_gb=hw['storage_gb'],
            battery_pct=hw['battery_pct'], bandwidth_mbps=net['bandwidth_mbps'], latency_ms=net['latency_ms'],
            data_size=int(rng.integers(100, 3000)), local_loss=dloss, trust_score=trust
        )
        profile.compute_scores()
        clients.append(profile)
    return clients

# ══════════════════════════════════════════════════════════════════════════════
# 2.  ABAC — Attribute-Based Access Control
# ══════════════════════════════════════════════════════════════════════════════

class ABACController:
    def __init__(self, min_ram_gb=2.0, min_storage_gb=4.0, min_bandwidth=5.0, min_trust_score=0.4):
        self.min_ram_gb      = min_ram_gb
        self.min_storage_gb  = min_storage_gb
        self.min_bandwidth   = min_bandwidth
        self.min_trust_score = min_trust_score

    def authorize(self, client: ClientProfile) -> Tuple[bool, str]:
        if client.ram_gb < self.min_ram_gb:
            return False, f"RAM {client.ram_gb:.1f}GB < {self.min_ram_gb}GB"
        if client.storage_gb < self.min_storage_gb:
            return False, f"Storage {client.storage_gb:.1f}GB < {self.min_storage_gb}GB"
        if client.bandwidth_mbps < self.min_bandwidth:
            return False, f"Bandwidth {client.bandwidth_mbps:.1f}Mbps < {self.min_bandwidth}Mbps"
        if client.trust_score < self.min_trust_score:
            return False, f"Trust {client.trust_score:.2f} < {self.min_trust_score}"
        return True, "AUTHORIZED"

    def filter_clients(self, clients: List[ClientProfile]) -> List[ClientProfile]:
        authorized = [c for c in clients if self.authorize(c)[0]]
        print(f"[ABAC] {len(authorized)}/{len(clients)} clients authorized")
        return authorized
    def update_trust(self, client: ClientProfile, success: bool, penalty: float = 0.1):
        
        if success:
            # Agar training successful rahi toh trust thoda badhao
            client.trust_score = min(1.0, client.trust_score + 0.02)
        else:
            # Agar client ne gadbad ki toh trust kam karo
            client.trust_score = max(0.0, client.trust_score - penalty)
        
        # Scores ko re-calculate karein taaki g_T update ho jaye
        client.compute_scores()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  PROMETHEE — Multi-Criteria Client Selection with Fairness Update
# ══════════════════════════════════════════════════════════════════════════════

class PROMETHEESelector:
    def __init__(self, weights=None, q=0.01, p=0.5):
        self.weights = weights or {'H': 0.25, 'N': 0.25, 'Q': 0.25, 'T': 0.25}
        self.q = q
        self.p = p

    def _compute_unicriterion_flows_sliding(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(scores)
        order = np.argsort(scores)
        sorted_s = scores[order]
        phi_pos = np.zeros(n)
        phi_neg = np.zeros(n)
        
        # Positive flow calculation
        win_sum, start, end = 0.0, 0, 0
        for i in range(n):
            l, u = sorted_s[i] - self.p, sorted_s[i] - self.q
            while start < i and sorted_s[start] < l:
                win_sum -= sorted_s[start]
                start += 1
            while end < n and sorted_s[end] <= u:
                win_sum += sorted_s[end]
                end += 1
            win_size = end - start
            if win_size > 0 and (self.p - self.q) > 0:
                phi_pos[order[i]] = ((win_size * (sorted_s[i] - self.q) - win_sum) / ((self.p - self.q) * max(n - 1, 1)))

        # Negative flow calculation
        neg_scores = -scores
        neg_order = np.argsort(neg_scores)
        neg_sorted = neg_scores[neg_order]
        win_sum, start, end = 0.0, 0, 0
        for i in range(n):
            l, u = neg_sorted[i] - self.p, neg_sorted[i] - self.q
            while start < i and neg_sorted[start] < l:
                win_sum -= neg_sorted[start]
                start += 1
            while end < n and neg_sorted[end] <= u:
                win_sum += neg_sorted[end]
                end += 1
            win_size = end - start
            if win_size > 0 and (self.p - self.q) > 0:
                phi_neg[neg_order[i]] = ((win_size * (neg_sorted[i] - self.q) - win_sum) / ((self.p - self.q) * max(n - 1, 1)))
        
        return phi_pos, phi_neg

    def rank_clients(self, clients: List[ClientProfile]) -> List[ClientProfile]:
        n = len(clients)
        if n == 0: return []
        criteria_scores = {'H': np.array([c.g_H for c in clients]), 'N': np.array([c.g_N for c in clients]),
                           'Q': np.array([c.g_Q for c in clients]), 'T': np.array([c.g_T for c in clients])}
        phi_plus, phi_minus = np.zeros(n), np.zeros(n)
        for crit, scores in criteria_scores.items():
            w = self.weights.get(crit, 0.25)
            pos, neg = self._compute_unicriterion_flows_sliding(scores)
            phi_plus += w * pos
            phi_minus += w * neg
        net_flow = phi_plus - phi_minus
        ranked_idx = np.argsort(-net_flow)
        return [clients[i] for i in ranked_idx]

    def select_clients(self, clients: List[ClientProfile], k: int, budget: Optional[float] = None) -> List[ClientProfile]:
        """UPDATED: Fair-SecureFedPROM Selection logic"""
        ranked = self.rank_clients(clients)
        
        # 90/10 Exploration-Exploitation strategy
        if random.random() > 0.10:
            pool = ranked
        else:
            print("\n[Fairness Event] Round uses Exploration! Including low-end/unauthorized devices...")
            pool = clients.copy()
            random.shuffle(pool)

        if budget is None: return pool[:k]
        selected, remaining = [], budget
        for c in pool:
            if len(selected) >= k: break
            if c.cost <= remaining:
                selected.append(c)
                remaining -= c.cost
        return selected

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING & EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def train_local(model, dataloader, optimizer, criterion, epochs=5, device="cpu"):
    model.train()
    model.to(device)
    total_loss, batches, start_time = 0.0, 0, time.time()
    for _ in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
    return total_loss / max(batches, 1), time.time() - start_time

def evaluate_global(model, dataloader, criterion, device="cpu"):
    model.eval()
    model.to(device)
    correct, total, total_loss, batches = 0, 0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            batches += 1
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1), total_loss / max(batches, 1)

def fedavg_aggregate(global_model, local_models, client_sizes):
    total_size = sum(client_sizes)
    global_sd = global_model.state_dict()
    for key in global_sd:
        weighted = torch.zeros_like(global_sd[key], dtype=torch.float32)
        for model, size in zip(local_models, client_sizes):
            weighted += (size / total_size) * model.state_dict()[key].float()
        global_sd[key] = weighted
    global_model.load_state_dict(global_sd)
    return global_model

def compute_communication_cost_mb(model, num_clients, num_rounds):
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return params_bytes * 2 * num_clients * num_rounds / 1e6