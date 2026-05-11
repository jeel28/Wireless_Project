
import os
import sys
import yaml
import itertools
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from server import run_experiment

DATASETS          = ["cifar10"]
CLIENT_COUNTS     = [10]
ALPHAS            = [0.5]
FRACTIONS         = [0.5]
STRAGGLER_PROBS   = [0.2]
NUM_ROUNDS    = 100
LOCAL_EPOCHS  = 5
BATCH_SIZE    = 32
LR            = 0.01
SEED          = 42



STRAG_DELAY   = [1, 3]    
RESULTS_DIR   = "results_cifar10_2run_0.5alpha"
COMBINED_DIR  = os.path.join(RESULTS_DIR, "combined")
CONFIGS_DIR   = "configs"

os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR,  exist_ok=True)


def make_config(dataset, n_clients, alpha, fraction, strag_prob) -> str:
    a_str    = str(alpha).replace(".", "p")
    f_str    = str(fraction).replace(".", "p")
    s_str    = str(strag_prob).replace(".", "p")
    run_tag  = f"fedavg_{dataset}_n{n_clients}_a{a_str}_f{f_str}_s{s_str}"
    cfg_path = os.path.join(CONFIGS_DIR, f"{run_tag}.yaml")

    cfg = {
        "dataset"        : dataset,
        "num_clients"    : n_clients,
        "num_rounds"     : NUM_ROUNDS,
        "alpha"          : alpha,
        "client_fraction": fraction,
        "local_epochs"   : LOCAL_EPOCHS,
        "batch_size"     : BATCH_SIZE,
        "lr"             : LR,
        "seed"           : SEED,
        "straggler_prob" : strag_prob,
        "straggler_delay": STRAG_DELAY if strag_prob > 0 else [0, 0],
        "results_dir"    : RESULTS_DIR,
        "run_tag"        : run_tag,
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path, run_tag

def _load_csv(run_tag):
    """Return DataFrame or None."""
    path = os.path.join(RESULTS_DIR, run_tag, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def plot_iid_vs_noniid(all_rows, dataset="mnist", n_clients=10, fraction=0.5):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    alpha_style = {
        "0.01": ("Non-IID α=0.01", "--"),
        "0.1":  ("Non-IID α=0.1",  "-."),
        "0.5":  ("Non-IID α=0.5",  ":"),
        "1.0":  ("Non-IID α=1.0",  (0, (3, 1, 1, 1))),
        "iid":  ("IID",            "-"),
    }
    plotted = 0
    for alpha, (label, ls) in alpha_style.items():
        a_str = alpha.replace(".", "p")
        f_str = str(fraction).replace(".", "p")
        tag   = f"fedavg_{dataset}_n{n_clients}_a{a_str}_f{f_str}_s0p0"
        df    = _load_csv(tag)
        if df is None:
            continue
        ax.plot(df["Round"], df["Test Accuracy"] * 100,
                ls=ls, lw=1.8, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Global test accuracy (%)", fontsize=12)
    ax.set_title(f"IID vs Non-IID  —  {dataset.upper()}, {n_clients} clients", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(COMBINED_DIR, f"iid_vs_noniid_{dataset}_n{n_clients}.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved → {out}")


def plot_client_scale(all_rows, dataset="mnist", alpha="0.1", fraction=0.5):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    plotted = 0
    for n in [10, 50, 100]:
        a_str = str(alpha).replace(".", "p")
        f_str = str(fraction).replace(".", "p")
        tag   = f"fedavg_{dataset}_n{n}_a{a_str}_f{f_str}_s0p0"
        df    = _load_csv(tag)
        if df is None:
            continue
        ax.plot(df["Round"], df["Test Accuracy"] * 100,
                lw=1.8, label=f"{n} clients")
        plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Global test accuracy (%)", fontsize=12)
    ax.set_title(f"Client count comparison  —  {dataset.upper()}, α={alpha}", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(COMBINED_DIR, f"client_scale_{dataset}_a{str(alpha).replace('.','p')}.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved → {out}")


def plot_straggler_effect(all_rows, dataset="mnist", n_clients=10, alpha="0.1", fraction=0.5):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    plotted = 0
    for sp in [0.0, 0.2, 0.4]:
        a_str = str(alpha).replace(".", "p")
        f_str = str(fraction).replace(".", "p")
        s_str = str(sp).replace(".", "p")
        tag   = f"fedavg_{dataset}_n{n_clients}_a{a_str}_f{f_str}_s{s_str}"
        df    = _load_csv(tag)
        if df is None:
            continue
        ax.plot(df["Round"], df["Test Accuracy"] * 100,
                lw=1.8, label=f"Straggler p={sp}")
        plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Global test accuracy (%)", fontsize=12)
    ax.set_title(f"Straggler effect  —  {dataset.upper()}, {n_clients} clients", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(COMBINED_DIR, f"straggler_{dataset}_n{n_clients}.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved → {out}")


def plot_fraction_effect(all_rows, dataset="mnist", n_clients=10, alpha="0.1"):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    plotted = 0
    for frac in [0.1, 0.3, 0.5]:
        a_str = str(alpha).replace(".", "p")
        f_str = str(frac).replace(".", "p")
        tag   = f"fedavg_{dataset}_n{n_clients}_a{a_str}_f{f_str}_s0p0"
        df    = _load_csv(tag)
        if df is None:
            continue
        ax.plot(df["Round"], df["Test Accuracy"] * 100,
                lw=1.8, label=f"Fraction={frac}")
        plotted += 1
    if plotted == 0:
        plt.close(fig); return
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Global test accuracy (%)", fontsize=12)
    ax.set_title(f"Participation fraction  —  {dataset.upper()}, {n_clients} clients", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(COMBINED_DIR, f"fraction_{dataset}_n{n_clients}.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved → {out}")

def build_pdf(df: pd.DataFrame, path: str):
    try:
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib       import colors
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.lib.styles   import getSampleStyleSheet
        from reportlab.lib.units    import cm
    except ImportError:
        print("  [PDF] reportlab not installed — skipping PDF. Run: pip install reportlab")
        return

    doc = SimpleDocTemplate(path, pagesize=landscape(A4),
                            leftMargin=1*cm, rightMargin=1*cm,
                            topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    elems  = [Paragraph("FedAvg Baseline — Summary of All Experiments", styles["Title"]),
              Spacer(1, 0.3*cm)]

    cols = ["run", "dataset", "num_clients", "alpha", "fraction",
            "straggler_prob", "final_accuracy", "convergence_round",
            "total_comm_mb", "avg_round_time"]
    cols = [c for c in cols if c in df.columns]
    sub  = df[cols].copy()
    sub["final_accuracy"] = (sub["final_accuracy"] * 100).round(2).astype(str) + "%"

    data = [list(sub.columns)] + sub.values.tolist()
    tbl  = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1,  0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0, 0), (-1,  0), colors.white),
        ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F3F4")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
    ]))
    elems.append(tbl)
    doc.build(elems)
    print(f"  PDF  → {path}")

def main():
    all_rows = []
    combos = list(itertools.product(
        DATASETS, CLIENT_COUNTS, ALPHAS, FRACTIONS, STRAGGLER_PROBS
    ))
    total = len(combos)
    print(f"\nTotal experiments planned: {total}")
    print("(Each takes ~1-3 min on CPU. Total est: {:.0f}-{:.0f} min)\n".format(
        total * 1, total * 3))

    for i, (dataset, n_clients, alpha, fraction, strag_prob) in enumerate(combos, 1):
        print(f"\n[{i}/{total}]", end=" ")
        cfg_path, run_tag = make_config(dataset, n_clients, alpha, fraction, strag_prob)
        done_flag = os.path.join(RESULTS_DIR, run_tag, "metrics.csv")
        if os.path.exists(done_flag):
            print(f"SKIP (already done): {run_tag}")
            try:
                df_ex = pd.read_csv(done_flag)
                all_rows.append({
                    "run"           : run_tag,
                    "dataset"       : dataset,
                    "num_clients"   : n_clients,
                    "alpha"         : str(alpha),
                    "fraction"      : fraction,
                    "straggler_prob": strag_prob,
                    "final_accuracy": df_ex["Test Accuracy"].iloc[-1],
                    "convergence_round": int(df_ex.loc[df_ex["Test Accuracy"] >= 0.8, "Round"].min())
                                         if (df_ex["Test Accuracy"] >= 0.8).any() else None,
                    "total_comm_mb" : df_ex["Comm Cost (MB)"].sum(),
                    "avg_round_time": df_ex["Round Time (s)"].mean(),
                })
            except Exception:
                pass
            continue

        try:
            summary = run_experiment(cfg_path)
            all_rows.append(summary)
        except Exception as e:
            print(f"  ERROR in {run_tag}: {e}")
            continue

    if all_rows:
        master_df = pd.DataFrame(all_rows)
        csv_path  = os.path.join(COMBINED_DIR, "final_results.csv")
        master_df.to_csv(csv_path, index=False)
        print(f"\nMaster CSV → {csv_path}")
        pdf_path = os.path.join(COMBINED_DIR, "final_results.pdf")
        build_pdf(master_df, pdf_path)
        print("\nGenerating combined comparison plots...")
        for ds in DATASETS:
            plot_iid_vs_noniid(all_rows, dataset=ds, n_clients=10,   fraction=0.5)
            plot_client_scale (all_rows, dataset=ds, alpha="0.1",    fraction=0.5)
            plot_straggler_effect(all_rows, dataset=ds, n_clients=10, alpha="0.1", fraction=0.5)
            plot_fraction_effect (all_rows, dataset=ds, n_clients=10, alpha="0.1")

        print("\nAll done.")
        print(f"  Combined outputs in: {COMBINED_DIR}/")
    else:
        print("No results to aggregate.")


if __name__ == "__main__":
    main()
