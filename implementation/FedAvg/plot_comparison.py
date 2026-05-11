import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


# =========================================================================== #
#  Helpers
# =========================================================================== #

def model_size_mb(parameters: List[np.ndarray]) -> float:
    return sum(p.nbytes for p in parameters) / (1024 ** 2)


# =========================================================================== #
#  MetricsLogger
# =========================================================================== #

class MetricsLogger:
    """
    Tracks per-round metrics and produces all mandatory plots + CSV.

    Per-round fields logged
    -----------------------
    Round | Test Accuracy | Test Loss | Comm Cost (MB) | Elapsed (s)
    Round Time (s) | Straggler Ratio | Participation Rate | Train Loss | Train Acc
    """

    def __init__(self, save_dir: str = "results"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # universal
        self.rounds        : List[int]   = []
        self.accuracies    : List[float] = []
        self.losses        : List[float] = []
        self.comm_costs    : List[float] = []
        self.timestamps    : List[float] = []

        # category 7
        self.round_times      : List[float] = []
        self.straggler_ratios : List[float] = []
        self.participation    : List[float] = []

        # training-side (from fit metrics)
        self.train_losses     : List[float] = []
        self.train_accs       : List[float] = []

        self.start_time = time.time()
        self.convergence_round: Optional[int] = None
        self.convergence_threshold = 0.80

    # ----------------------------------------------------------------------- #

    def log(
        self,
        rnd           : int,
        accuracy      : float,
        loss          : float,
        comm_mb       : float,
        round_time    : float = 0.0,
        straggler_ratio: float = 0.0,
        participation : float = 0.0,
        train_loss    : float = 0.0,
        train_acc     : float = 0.0,
    ):
        self.rounds.append(rnd)
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        self.comm_costs.append(comm_mb)
        self.timestamps.append(time.time() - self.start_time)
        self.round_times.append(round_time)
        self.straggler_ratios.append(straggler_ratio)
        self.participation.append(participation)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)

        if self.convergence_round is None and accuracy >= self.convergence_threshold:
            self.convergence_round = rnd
            print(f"  [Convergence] {self.convergence_threshold*100:.0f}% reached at round {rnd}")

    # ----------------------------------------------------------------------- #

    def save_csv(self, filename: str = "metrics.csv"):
        path = os.path.join(self.save_dir, filename)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Round", "Test Accuracy", "Test Loss",
                "Comm Cost (MB)", "Elapsed (s)",
                "Round Time (s)", "Straggler Ratio", "Participation Rate",
                "Train Loss", "Train Accuracy",
            ])
            rows = zip(
                self.rounds, self.accuracies, self.losses,
                self.comm_costs, self.timestamps,
                self.round_times, self.straggler_ratios, self.participation,
                self.train_losses, self.train_accs,
            )
            for row in rows:
                w.writerow([round(float(v), 6) for v in row])
        print(f"  CSV  → {path}")
        return path

    # ----------------------------------------------------------------------- #
    # Plots
    # ----------------------------------------------------------------------- #

    def _save(self, fig, filename: str):
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → {path}")
        return path

    def plot_accuracy(self, label: str = "FedAvg", filename: str = "accuracy.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.rounds, [a * 100 for a in self.accuracies],
                marker="o", ms=3, lw=1.8, label=label)
        if self.convergence_round:
            ax.axvline(self.convergence_round, color="red", ls="--", alpha=0.7,
                       label=f"Convergence @ round {self.convergence_round}")
        ax.set_xlabel("Communication round", fontsize=12)
        ax.set_ylabel("Global test accuracy (%)", fontsize=12)
        ax.set_title("Global accuracy vs communication rounds", fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        return self._save(fig, filename)

    def plot_loss(self, label: str = "FedAvg", filename: str = "loss.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.rounds, self.losses,
                marker="s", ms=3, lw=1.8, color="coral", label=label)
        ax.set_xlabel("Communication round", fontsize=12)
        ax.set_ylabel("Global test loss", fontsize=12)
        ax.set_title("Global loss vs communication rounds", fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        return self._save(fig, filename)

    def plot_round_time(self, filename: str = "time.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.rounds, self.round_times, marker="^", ms=3, lw=1.8, color="steelblue")
        ax.set_xlabel("Communication round", fontsize=12)
        ax.set_ylabel("Round time (s)", fontsize=12)
        ax.set_title("Round completion time vs rounds", fontsize=13)
        ax.grid(True, alpha=0.3)
        return self._save(fig, filename)

    def plot_straggler(self, filename: str = "straggler.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.rounds, self.straggler_ratios,
                marker="D", ms=3, lw=1.8, color="tomato")
        ax.set_xlabel("Communication round", fontsize=12)
        ax.set_ylabel("Straggler ratio", fontsize=12)
        ax.set_title("Straggler ratio per round", fontsize=13)
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        return self._save(fig, filename)

    def save_all_plots(self, label: str = "FedAvg"):
        self.plot_accuracy(label=label)
        self.plot_loss(label=label)
        self.plot_round_time()
        self.plot_straggler()

    # ----------------------------------------------------------------------- #

    def summary(self) -> Dict:
        return {
            "final_accuracy"   : round(self.accuracies[-1], 6)  if self.accuracies   else 0.0,
            "final_loss"       : round(self.losses[-1], 6)      if self.losses        else 0.0,
            "convergence_round": self.convergence_round,
            "total_comm_mb"    : round(sum(self.comm_costs), 4),
            "avg_round_time"   : round(float(np.mean(self.round_times)), 4) if self.round_times else 0.0,
            "avg_straggler"    : round(float(np.mean(self.straggler_ratios)), 4) if self.straggler_ratios else 0.0,
        }
