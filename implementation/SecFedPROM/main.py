"""
main.py — SecureFedPROM Simulation Runner
==========================================
Simulates federated learning WITHOUT needing a live Flower server/client process.
This is a self-contained simulation suitable for academic experiments.

Usage:
    python main.py --dataset fmnist --num_clients 10 --rounds 50 --alpha 0.1
    python main.py --dataset cifar10 --num_clients 50 --rounds 50 --alpha 0.5
    python main.py --dataset fmnist --num_clients 100 --rounds 30 --alpha IID
    python main.py --run_all   # runs full comparison of all strategies
"""

import argparse
import copy
import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model, get_model_parameters, set_model_parameters
from data import partition_data
# Purane import ko hata kar sirf itna likhein:
from utils import (
    simulate_clients,
    ABACController,
    PROMETHEESelector,
    train_local,
    evaluate_global,
    fedavg_aggregate,
    compute_communication_cost_mb
)

# ─── Fixed seeds (professor requirement) ─────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FL SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_fl_simulation(dataset:    str,
                      num_clients: int,
                      num_rounds:  int,
                      alpha:       float,
                      strategy:    str,
                      clients_per_round: int,
                      budget:      float = 50.0,
                      local_epochs: int  = 5,
                      batch_size:   int  = 32,
                      lr:           float = 0.01,
                      target_accuracy: float = 80.0,
                      device:       str  = "cpu") -> dict:
    """
    Full FL simulation loop.

    Returns a dict with:
        round_accuracies, round_losses, convergence_round,
        toa_50, toa_60, toa_70, final_accuracy,
        comm_cost_mb, total_time_s, strategy
    """
    print(f"\n{'='*60}")
    print(f"  Strategy : {strategy.upper()}")
    print(f"  Dataset  : {dataset}  |  Clients: {num_clients}  |  α={alpha}")
    print(f"  Rounds   : {num_rounds}  |  Budget: {budget}")
    print(f"{'='*60}")

    # 1. Load and partition data
    client_loaders, test_loader, num_classes = partition_data(
        dataset, num_clients, alpha, batch_size
    )

    # 2. Simulate client hardware/network profiles
    client_profiles = simulate_clients(num_clients, mislabel_fraction=0.07)
    for i, profile in enumerate(client_profiles):
        profile.data_size = len(client_loaders[i].dataset)
        profile.compute_scores()

    # 3. ABAC filter (zero-trust gate)
    abac      = ABACController()
    authorized = abac.filter_clients(client_profiles)
    auth_ids   = {c.client_id for c in authorized}

    # 4. PROMETHEE selector
    promethee = PROMETHEESelector()

    # 5. Global model
    global_model = get_model(dataset, num_classes).to(device)
    criterion    = nn.CrossEntropyLoss()

    # 6. Metrics storage
    round_accuracies = []
    round_losses     = []
    convergence_round = None
    toa = {50: None, 60: None, 70: None}
    total_start = time.time()
    cumulative_time = 0.0

    # 7. Training loop
    for rnd in range(1, num_rounds + 1):
        rnd_start = time.time()

        # --- Client selection ---
        eligible = [p for p in authorized]  # ABAC-filtered pool

        k = min(clients_per_round, len(eligible))

        if strategy == "securefedprom":
            selected = promethee.select_clients(eligible, k, budget)
        elif strategy == "random":
            selected = random_selection(eligible, k)
        elif strategy == "power_of_choice":
            selected = power_of_choice_selection(eligible, k)
        elif strategy == "greedy":
            selected = greedy_selection(eligible, k)
        elif strategy == "resource_aware":
            selected = resource_aware_selection(eligible, k)
        elif strategy == "price_first":
            selected = price_first_selection(eligible, k, budget)
        else:
            selected = random_selection(eligible, k)

        selected_ids = [c.client_id for c in selected]

        # --- Local training ---
        local_models = []
        local_sizes  = []

        for cid in selected_ids:
            local_model = copy.deepcopy(global_model)
            optimizer   = optim.SGD(local_model.parameters(),
                                    lr=lr, momentum=0.9, weight_decay=1e-4)
            avg_loss, _ = train_local(
                local_model, client_loaders[cid],
                optimizer, criterion,
                epochs=local_epochs, device=device
            )
            # Update data quality (local loss) for PROMETHEE
            client_profiles[cid].local_loss = avg_loss
            client_profiles[cid].compute_scores()
            abac.update_trust(client_profiles[cid], success=True)

            local_models.append(local_model)
            local_sizes.append(len(client_loaders[cid].dataset))

        # --- Aggregation (FedAvg, Eq. 2) ---
        global_model = fedavg_aggregate(global_model, local_models, local_sizes)

        # --- Global evaluation ---
        accuracy, loss = evaluate_global(global_model, test_loader,
                                          criterion, device)
        round_accuracies.append(accuracy)
        round_losses.append(loss)

        rnd_elapsed = time.time() - rnd_start
        cumulative_time += rnd_elapsed

        # --- ToA tracking ---
        for threshold in [50, 60, 70]:
            if toa[threshold] is None and accuracy >= threshold:
                toa[threshold] = cumulative_time

        # --- Convergence round ---
        if convergence_round is None and accuracy >= target_accuracy:
            convergence_round = rnd

        print(f"  Round {rnd:3d}/{num_rounds} | "
              f"Acc: {accuracy:6.2f}% | Loss: {loss:.4f} | "
              f"Clients: {len(selected_ids)} | "
              f"Time: {rnd_elapsed:.1f}s")

    total_time = time.time() - total_start
    comm_cost  = compute_communication_cost_mb(global_model, clients_per_round, num_rounds)

    return {
        "strategy":           strategy,
        "dataset":            dataset,
        "num_clients":        num_clients,
        "num_rounds":         num_rounds,
        "alpha":              alpha,
        "round_accuracies":   round_accuracies,
        "round_losses":       round_losses,
        "final_accuracy":     round_accuracies[-1] if round_accuracies else 0.0,
        "convergence_round":  convergence_round,
        "toa_50":             toa[50],
        "toa_60":             toa[60],
        "toa_70":             toa[70],
        "comm_cost_mb":       comm_cost,
        "total_time_s":       total_time
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING  (professor mandatory plots)
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "securefedprom":  "#6C3483",
    "random":         "#E74C3C",
    "power_of_choice":"#2196F3",
    "greedy":         "#4CAF50",
    "resource_aware": "#FF9800",
    "price_first":    "#9C27B0"
}
LABELS = {
    "securefedprom":  "SecureFedPROM (Ours)",
    "random":         "Random Selection",
    "power_of_choice":"Power of Choice",
    "greedy":         "Greedy",
    "resource_aware": "Resource Aware",
    "price_first":    "Price First"
}


def plot_accuracy_vs_rounds(all_results: list, title_suffix: str = ""):
    """Figure 1: Global accuracy vs communication rounds (mandatory)."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for res in all_results:
        strat = res["strategy"]
        ax.plot(range(1, len(res["round_accuracies"]) + 1),
                res["round_accuracies"],
                label=LABELS.get(strat, strat),
                color=COLORS.get(strat, "black"),
                linewidth=2.2 if strat == "securefedprom" else 1.4,
                linestyle="-"  if strat == "securefedprom" else "--")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy vs Communication Rounds {title_suffix}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"accuracy_vs_rounds{title_suffix.replace(' ','_')}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


def plot_loss_vs_rounds(all_results: list, title_suffix: str = ""):
    """Figure 2: Global loss vs communication rounds (mandatory)."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for res in all_results:
        strat = res["strategy"]
        ax.plot(range(1, len(res["round_losses"]) + 1),
                res["round_losses"],
                label=LABELS.get(strat, strat),
                color=COLORS.get(strat, "black"),
                linewidth=2.2 if strat == "securefedprom" else 1.4,
                linestyle="-"  if strat == "securefedprom" else "--")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Test Loss", fontsize=12)
    ax.set_title(f"Loss vs Communication Rounds {title_suffix}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"loss_vs_rounds{title_suffix.replace(' ','_')}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


def plot_iid_vs_noniid(iid_res: dict, noniid_res: dict, dataset: str):
    """Figure 3: IID vs Non-IID comparison (mandatory)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for ax, res, label in zip(axes,
                               [iid_res, noniid_res],
                               ["IID", f"Non-IID (α={noniid_res['alpha']})"]):
        ax.plot(range(1, len(res["round_accuracies"]) + 1),
                res["round_accuracies"],
                color=COLORS["securefedprom"], linewidth=2.2)
        ax.set_title(f"SecureFedPROM — {label}\n{dataset.upper()}", fontsize=12)
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel("Global Test Accuracy (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    plt.suptitle("IID vs Non-IID Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"iid_vs_noniid_{dataset}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


def plot_final_accuracy_bar(all_results: list, title_suffix: str = ""):
    """Figure 4: Baseline vs Proposed — bar chart (mandatory)."""
    strategies = [r["strategy"] for r in all_results]
    accuracies = [r["final_accuracy"] for r in all_results]
    colors     = [COLORS.get(s, "#607D8B") for s in strategies]
    labels     = [LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.bar(labels, accuracies, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    # Highlight SecureFedPROM
    for i, (bar, strat) in enumerate(zip(bars, strategies)):
        if strat == "securefedprom":
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{accuracies[i]:.1f}%", ha="center", va="bottom", fontsize=9.5)

    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Final Accuracy Comparison {title_suffix}", fontsize=13)
    ax.tick_params(axis='x', labelsize=9, rotation=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylim(0, max(accuracies) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"final_accuracy_bar{title_suffix.replace(' ','_')}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE  (professor mandatory format)
# ══════════════════════════════════════════════════════════════════════════════

def save_results_table(all_results: list, filename: str = "results_table.csv"):
    """Save mandatory results table (professor format, Section 6.2)."""
    rows = []
    for r in all_results:
        rows.append({
            "Method":             LABELS.get(r["strategy"], r["strategy"]),
            "Dataset":            r["dataset"].upper(),
            "#Clients":           r["num_clients"],
            "#Rounds":            r["num_rounds"],
            "Test Accuracy (%)":  f"{r['final_accuracy']:.2f}",
            "Convergence Round":  r["convergence_round"] or "N/A",
            "Comm. Cost (MB)":    f"{r['comm_cost_mb']:.1f}",
            "ToA@50 (s)":         f"{r['toa_50']:.1f}" if r['toa_50'] else "N/A",
            "ToA@60 (s)":         f"{r['toa_60']:.1f}" if r['toa_60'] else "N/A",
            "ToA@70 (s)":         f"{r['toa_70']:.1f}" if r['toa_70'] else "N/A",
        })

    df = pd.DataFrame(rows)
    # Bold best accuracy row
    best_idx = df["Test Accuracy (%)"].astype(float).idxmax()
    fpath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(fpath, index=False)
    print(f"\n[Table] Saved: {fpath}")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

ALL_STRATEGIES = ["securefedprom", "random", "power_of_choice",
                  "greedy", "resource_aware", "price_first"]


def main():
    parser = argparse.ArgumentParser(description="SecureFedPROM FL Simulation")
    parser.add_argument("--dataset",     type=str, default="fmnist",
                        choices=["fmnist", "cifar10"],
                        help="Dataset to use")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Number of FL clients (10, 50, 100)")
    parser.add_argument("--rounds",      type=int, default=30,
                        help="Number of FL communication rounds")
    parser.add_argument("--alpha",       type=str, default="0.5",
                        help="Dirichlet alpha (0.01, 0.1, 0.5, 1.0, IID)")
    parser.add_argument("--strategy",    type=str, default="securefedprom",
                        choices=ALL_STRATEGIES + ["all"],
                        help="Client selection strategy")
    parser.add_argument("--clients_per_round", type=int, default=5,
                        help="Clients selected per round (paper uses 20, reduce for speed)")
    parser.add_argument("--budget",      type=float, default=50.0,
                        help="Total budget for client selection")
    parser.add_argument("--run_all",     action="store_true",
                        help="Run full comparison of all strategies")
    parser.add_argument("--iid_compare", action="store_true",
                        help="Also run IID experiment for comparison plot")
    args = parser.parse_args()

    # Parse alpha
    alpha = float('inf') if args.alpha.upper() == "IID" else float(args.alpha)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")

    all_results = []

    if args.run_all or args.strategy == "all":
        # Full comparison — run every strategy
        for strat in ALL_STRATEGIES:
            res = run_fl_simulation(
                dataset         = args.dataset,
                num_clients     = args.num_clients,
                num_rounds      = args.rounds,
                alpha           = alpha,
                strategy        = strat,
                clients_per_round = args.clients_per_round,
                budget          = args.budget,
                device          = device
            )
            all_results.append(res)

        suffix = f"_{args.dataset}_n{args.num_clients}_a{args.alpha}_r{args.rounds}"
        plot_accuracy_vs_rounds(all_results, suffix)
        plot_loss_vs_rounds(all_results, suffix)
        plot_final_accuracy_bar(all_results, suffix)
        df = save_results_table(all_results, f"results{suffix}.csv")

        # IID vs Non-IID comparison
        if args.iid_compare and alpha != float('inf'):
            iid_res = run_fl_simulation(
                dataset=args.dataset, num_clients=args.num_clients,
                num_rounds=args.rounds, alpha=float('inf'),
                strategy="securefedprom",
                clients_per_round=args.clients_per_round,
                budget=args.budget, device=device
            )
            noniid_res = [r for r in all_results
                          if r["strategy"] == "securefedprom"][0]
            plot_iid_vs_noniid(iid_res, noniid_res, args.dataset)

    else:
        # Single strategy run
        res = run_fl_simulation(
            dataset         = args.dataset,
            num_clients     = args.num_clients,
            num_rounds      = args.rounds,
            alpha           = alpha,
            strategy        = args.strategy,
            clients_per_round = args.clients_per_round,
            budget          = args.budget,
            device          = device
        )
        all_results.append(res)
        suffix = f"_{args.dataset}_{args.strategy}_n{args.num_clients}_a{args.alpha}"
        plot_accuracy_vs_rounds(all_results, suffix)
        plot_loss_vs_rounds(all_results, suffix)
        save_results_table(all_results, f"results{suffix}.csv")

    # Save raw JSON
    json_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items()
                    if k not in ["round_accuracies", "round_losses"]}
                   for r in all_results], f, indent=2)

    print(f"\n✅ All done! Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
