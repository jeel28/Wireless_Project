# SecureFedPROM — Federated Learning Term Paper Implementation

> **Course**: Federated Learning — BDS Jan-April 2026  
> **Paper**: *SecureFedPROM: A Zero-Trust Federated Learning Approach With Multi-Criteria Client Selection* (IEEE JSAC 2025)  
> **Category**: Category 7 – System Heterogeneity + Security

---

## 📁 Repository Structure

```
securefedprom/
├── configs/                  # YAML experiment configs
│   ├── fmnist_10clients.yaml
│   └── cifar10_50clients.yaml
├── src/                      # All source code
│   ├── model.py              # CNN architectures (FMNIST, CIFAR-10)
│   ├── data.py               # Data loading + Dirichlet Non-IID partitioning
│   ├── utils.py              # ABAC + PROMETHEE + selection strategies
│   ├── client.py             # Flower FL client
│   ├── server.py             # Flower FL server + SecureFedPROM strategy
│   └── main.py               # Main simulation runner ← START HERE
├── results/                  # Auto-generated plots and CSV tables
├── report/                   # Final PDF report
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd securefedprom

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

### Quick test (10 clients, 10 rounds)
```bash
cd src
python main.py --dataset fmnist --num_clients 10 --rounds 10 --alpha 0.5
```

### Full comparison of all strategies (as in paper)
```bash
python main.py --dataset fmnist --num_clients 10 --rounds 50 --alpha 0.1 --run_all
```

### IID vs Non-IID comparison
```bash
python main.py --dataset fmnist --num_clients 10 --rounds 30 --alpha 0.5 --run_all --iid_compare
```

### CIFAR-10 with 50 clients
```bash
python main.py --dataset cifar10 --num_clients 50 --rounds 50 --alpha 0.5 --run_all
```

### All alpha values (professor requirement)
```bash
for alpha in 0.01 0.1 0.5 1.0 IID; do
  python main.py --dataset fmnist --num_clients 10 --rounds 30 --alpha $alpha --run_all
done
```

---

## 📊 What Gets Generated

After running, check the `results/` folder:

| File | Description |
|------|-------------|
| `accuracy_vs_rounds_*.png` | Figure 1 (mandatory) |
| `loss_vs_rounds_*.png` | Figure 2 (mandatory) |
| `iid_vs_noniid_*.png` | Figure 3 (mandatory) |
| `final_accuracy_bar_*.png` | Figure 4 — bar chart comparison |
| `results_*.csv` | Mandatory results table |
| `raw_results.json` | Full metrics in JSON |

---

## 🔬 Key Algorithms Implemented

### 1. ABAC (Attribute-Based Access Control)
```
Client requests to join → ABAC checks:
  ✓ RAM ≥ 2GB
  ✓ Storage ≥ 4GB  
  ✓ Bandwidth ≥ 5Mbps
  ✓ Trust Score ≥ 0.4
→ If pass: issued PKI certificate → eligible for training
→ If fail: rejected from pool
```

### 2. PROMETHEE Ranking (Algorithm 2)
```
For each client, compute attribute vector V_i = [g_H, g_N, g_Q, g_T]
For each criterion: sort + sliding window → positive/negative flows
Aggregate flows with weights → net outranking score φ(c)
Select top-K clients by φ(c) within budget B
```

### 3. FedAvg Aggregation (Eq. 2)
```
w_global = (1/N_S) * Σ n_i * w_i
```

---

## 📋 Baseline Strategies Compared

| Strategy | Description |
|----------|-------------|
| **SecureFedPROM** | ABAC + PROMETHEE multi-criteria (proposed) |
| Random | Random client sampling |
| Power of Choice | Highest validation loss clients |
| Greedy | Best data_size/cost ratio |
| Resource Aware | Best hardware + network score |
| Price First | Cheapest clients within budget |

---

## 📈 Expected Results (from paper)

| Method | FMNIST Final Acc | ToA@50% (sec) |
|--------|-----------------|----------------|
| Random | 75.55% | 530s |
| Power of Choice | 61.19% | 312s |
| Greedy | 75.55% | 718s |
| **SecureFedPROM** | **80.60%** | **56.41s** |

---

## 🔒 Seeds (Professor Requirement)
All seeds fixed at 42:
```python
random.seed(42)
numpy.seed(42)
torch.manual_seed(42)
```

---

## 👥 Group Info
- **Group**: [Your Group Name]
- **Members**: [Names]
- **Category**: 7 – System Heterogeneity
- **YouTube Demo**: [Add link after recording]
