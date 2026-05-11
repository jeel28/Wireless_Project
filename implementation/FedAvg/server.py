
import os, sys, time, yaml
import torch
import flwr as fl
from flwr.common import Context
from collections import OrderedDict
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model  import get_model
from data   import load_dataset, dirichlet_partition, get_test_dataloader, set_seeds
from client import FLClient
from utils  import MetricsLogger, model_size_mb


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)
    """Aggregate train_loss, train_accuracy, straggler_ratio from all clients."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        "train_loss"     : sum(n * m.get("train_loss",     0.0) for n, m in metrics) / total,
        "train_accuracy" : sum(n * m.get("train_accuracy", 0.0) for n, m in metrics) / total,
        "straggler_ratio": sum(    m.get("straggler",      0  ) for _, m in metrics) / len(metrics),
    }


def _eval_metrics_agg(metrics):
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {"accuracy": sum(n * m["accuracy"] for n, m in metrics) / total}

def build_evaluate_fn(model_fn, test_loader, device):
    import torch.nn as nn

    def evaluate(server_round, parameters, config):
        model = model_fn()
        sd    = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
        )
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()

        criterion             = nn.CrossEntropyLoss()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y    = X.to(device), y.to(device)
                logits  = model(X)
                total_loss += criterion(logits, y).item() * len(y)
                correct    += (logits.argmax(1) == y).sum().item()
                n          += len(y)
        return total_loss / max(n, 1), {"accuracy": correct / max(n, 1)}

    return evaluate

def run_experiment(config_path: str) -> Dict:
    cfg = load_config(config_path)
    set_seeds(cfg.get("seed", 42))

    dataset_name    = cfg["dataset"]
    num_clients     = int(cfg["num_clients"])
    num_rounds      = int(cfg["num_rounds"])
    alpha           = cfg["alpha"]
    fraction        = float(cfg.get("client_fraction", 0.5))
    local_epochs    = int(cfg.get("local_epochs", 5))
    batch_size      = int(cfg.get("batch_size", 32))
    lr              = float(cfg.get("lr", 0.01))
    seed            = int(cfg.get("seed", 42))
    straggler_prob  = float(cfg.get("straggler_prob", 0.0))
    straggler_delay = cfg.get("straggler_delay", [0, 0])
    base_results    = cfg.get("results_dir", "results")
    run_tag         = cfg["run_tag"]

    # per-run subfolder
    run_dir = os.path.join(base_results, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RUN : {run_tag}")
    print(f"  ds={dataset_name} | clients={num_clients} | rounds={num_rounds} | alpha={alpha}")
    print(f"  fraction={fraction} | straggler_prob={straggler_prob}")
    print(f"{'='*60}")

    # ---- data ----
    train_ds, test_ds = load_dataset(dataset_name)
    alpha_val         = float("inf") if str(alpha).lower() == "iid" else float(alpha)
    client_indices    = dirichlet_partition(train_ds, num_clients, alpha_val, seed)
    test_loader       = get_test_dataloader(test_ds)

    # ---- model / comm cost ----
    init_model    = get_model(dataset_name)
    init_params   = [v.cpu().numpy() for v in init_model.state_dict().values()]
    model_mb      = model_size_mb(init_params)
    # upload + download per round
    comm_per_round = model_mb * 2 * max(1, int(num_clients * fraction))

    # ---- logger ----
    logger = MetricsLogger(run_dir)

    # ---- strategy ----
    strategy = fl.server.strategy.FedAvg(
        fraction_fit                    = fraction,
        fraction_evaluate               = 0.0,
        min_fit_clients                 = max(1, int(num_clients * fraction)),
        min_available_clients           = num_clients,
        evaluate_fn                     = build_evaluate_fn(
                                              lambda: get_model(dataset_name),
                                              test_loader, device
                                          ),
        fit_metrics_aggregation_fn      = _fit_metrics_agg,
        evaluate_metrics_aggregation_fn = _eval_metrics_agg,
        initial_parameters              = fl.common.ndarrays_to_parameters(init_params),
    )

    # ---- client factory ----
    def client_fn(context: Context):
        cid = int(context.node_id) % num_clients
        return FLClient(
            cid            = cid,
            model          = get_model(dataset_name),
            train_indices  = client_indices[cid],
            train_dataset  = train_ds,
            device         = device,
            local_epochs   = local_epochs,
            batch_size     = batch_size,
            lr             = lr,
            seed           = seed,
            straggler_prob = straggler_prob,
            straggler_delay= straggler_delay,
        ).to_client()

    # ---- simulation ----
    t0      = time.time()
    history = fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = num_clients,
        config           = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy         = strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args    = {"include_dashboard": False, "log_to_driver": False},
    )
    total_wall = time.time() - t0
    losses_list = history.losses_centralized      # [(0, v), (1, v), ...]
    acc_list    = history.metrics_centralized.get("accuracy", [])

    # skip round-0 init entry if present
    losses_list = [x for x in losses_list if x[0] > 0]
    acc_list    = [x for x in acc_list    if x[0] > 0]

    # fit metrics (train_loss, train_accuracy, straggler_ratio)
    fit_train_loss  = history.metrics_distributed_fit.get("train_loss",      [])
    fit_train_acc   = history.metrics_distributed_fit.get("train_accuracy",  [])
    fit_strag       = history.metrics_distributed_fit.get("straggler_ratio", [])

    def _get(lst, rnd):
        for r, v in lst:
            if r == rnd:
                return v
        return 0.0

    avg_round_time = total_wall / max(num_rounds, 1)

    for rnd_entry, acc_entry in zip(losses_list, acc_list):
        rnd  = int(rnd_entry[0])
        loss = float(rnd_entry[1])
        acc  = float(acc_entry[1])
        logger.log(
            rnd              = rnd,
            accuracy         = acc,
            loss             = loss,
            comm_mb          = comm_per_round,
            round_time       = avg_round_time,
            straggler_ratio  = float(_get(fit_strag,      rnd)),
            participation    = fraction,
            train_loss       = float(_get(fit_train_loss, rnd)),
            train_acc        = float(_get(fit_train_acc,  rnd)),
        )

    # ---- save outputs ----
    csv_file = f"metrics.csv"
    logger.save_csv(csv_file)
    logger.save_all_plots(label=run_tag)

    summary = logger.summary()
    summary["run"]        = run_tag
    summary["dataset"]    = dataset_name
    summary["num_clients"]= num_clients
    summary["num_rounds"] = num_rounds
    summary["alpha"]      = str(alpha)
    summary["fraction"]   = fraction
    summary["straggler_prob"] = straggler_prob
    summary["run_dir"]    = run_dir

    print(f"\n  Summary: acc={summary['final_accuracy']:.4f} | "
          f"conv_round={summary['convergence_round']} | "
          f"comm={summary['total_comm_mb']:.1f} MB")
    return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python server.py configs/fedavg.yaml")
        sys.exit(1)
    run_experiment(sys.argv[1])
