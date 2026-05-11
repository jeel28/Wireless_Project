# Wireless Project — Federated Learning using FedAvg, Oort, and SecFedPROM

# # 👨‍💻 Authors

| Name | RollNo |
|---|---|
| Tirth Patel | 2511ai07 |
| Avni Verma | 2511ai08 |
| Jeel Patel | 2511ai12 |
| Shreya Singhal | 2511ai21 |
| Ved Parkash | 2511ai58 |
| Optimizer | SGD |
| Framework | Flower |

## 📌 Project Overview

This project implements and compares multiple Federated Learning (FL) strategies:

- **FedAvg (Traditional Federated Averaging)**
- **Oort (Guided Participant Selection for Efficient FL)**
- **SecFedPROM (Secure Federated Learning Framework)**

The implementation is built using:

- Python
- PyTorch
- Flower (FL Framework)
- Ray Simulation

Datasets used:

- MNIST
- FashionMNIST
- CIFAR-10

The project focuses on:

- Non-IID data partitioning
- Client selection strategies
- Communication efficiency
- Straggler handling
- Secure aggregation concepts
- Federated learning performance comparison

---

# 📂 Project Structure

```text
Wireless_Project/
│
├── implementation/
│   │
│   ├── FedAvg/
│   │   ├── client.py
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── plot_comparison.py
│   │   ├── run.py
│   │   ├── server.py
│   │   └── utils.py
│   │
│   ├── Oort/
│   │   └── oort_fl_fixed.py
│   │
│   └── SecFedPROM/
│       ├── client.py
│       ├── data.py
│       ├── main.py
│       ├── model.py
│       ├── server.py
│       ├── utils.py
│       ├── requirements.txt
│       └── README.md
│
├── config/
├── Results/
├── requirements.txt
├── wireless_paper_AI_report.pdf
└── Wireless_paper_Plag_report.pdf
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Wireless_Project.git
cd Wireless_Project
```

---

## 2️⃣ Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

Create a `requirements.txt` file with:

```txt
flwr[simulation]==1.8.0

torch==2.2.0
torchvision==0.17.0

numpy==1.26.4
pandas==2.2.0
matplotlib==3.8.4
seaborn==0.12.2
scikit-learn==1.4.2
scipy==1.11.4

cryptography==41.0.7
protobuf==5.29.1
pyyaml==6.0.1
tqdm==4.66.2

ray==2.10.0
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Information

Datasets are downloaded automatically using torchvision.

Supported datasets:

- MNIST
- FashionMNIST
- CIFAR-10

---

# 🚀 Running the Project

# 1️⃣ Running FedAvg

Move to FedAvg folder:

```bash
cd Wireless_Project/implementation/FedAvg
```

Run the experiment:

```bash
python run.py
```

This will:

- Start federated learning simulation
- Train multiple clients
- Aggregate using FedAvg
- Save metrics/results

---

## 📌 Important FedAvg Files

| File | Description |
|---|---|
| `client.py` | Client-side training logic |
| `server.py` | Federated server logic |
| `model.py` | Neural network models |
| `data.py` | Dataset loading and partitioning |
| `utils.py` | Helper functions |
| `plot_comparison.py` | Graph plotting and visualization |
| `run.py` | Main execution script |

---

# 2️⃣ Running Oort

Move to Oort folder:

```bash
cd Wireless_Project/implementation/Oort
```

Run:

```bash
python oort_fl_fixed.py
```

This implementation includes:

- Oort participant selection
- Straggler-aware scheduling
- Utility-based client scoring
- Non-IID partitioning
- Communication analysis

---

## 📌 Oort Features

- Guided participant selection
- Exploration vs exploitation strategy
- Pacer mechanism
- Straggler simulation
- Dirichlet partitioning
- Accuracy and communication tracking

---

# 3️⃣ Running SecFedPROM

Move to SecFedPROM folder:

```bash
cd Wireless_Project/implementation/SecFedPROM
```

Run:

```bash
python main.py
```

This module focuses on:

- Secure Federated Learning
- Privacy-aware aggregation
- Communication efficiency
- Robust client-server workflow

---

## 📌 Important SecFedPROM Files

| File | Description |
|---|---|
| `main.py` | Main execution file |
| `client.py` | Client logic |
| `server.py` | Server aggregation |
| `model.py` | Deep learning model |
| `data.py` | Dataset handling |
| `utils.py` | Utility functions |

---

# 📊 Graphs and Visualization

For FedAvg graph plotting:

```bash
python plot_comparison.py
```

This script generates:

- Accuracy graphs
- Loss graphs
- Strategy comparison plots
- Communication cost analysis

---

# 🧠 Federated Learning Concepts Used

## FedAvg

Traditional Federated Averaging algorithm where:

- Clients train locally
- Model updates are aggregated globally
- No guided participant selection

---

## Oort

Oort improves FL efficiency using:

- Utility-based client selection
- Straggler mitigation
- Exploration-exploitation balancing
- Faster convergence

---

## SecFedPROM

SecFedPROM focuses on:

- Security
- Privacy preservation
- Reliable aggregation
- Secure communication

---

# 🔬 Experiment Configuration

Typical experiment settings:

| Parameter | Value |
|---|---|
| Clients | 10 / 50 / 100 |
| Rounds | 50 / 100 |
| Dataset | MNIST / CIFAR-10 |
| Partitioning | IID / Non-IID |
| Alpha Values | 0.01, 0.1, 0.5, 1.0 |
| Optimizer | SGD |
| Framework | Flower |

---

# 📈 Outputs

Generated outputs may include:

- CSV result files
- Accuracy logs
- Loss values
- Communication statistics
- PNG graph plots

---

# 📚 Technologies Used

- Python
- PyTorch
- Flower
- Ray
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# ⭐ Recommended Workflow

## Run FedAvg

```bash
cd Wireless_Project/implementation/FedAvg
python run.py
```

## Run Oort

```bash
cd Wireless_Project/implementation/Oort
python oort_fl_fixed.py
```

## Run SecFedPROM

```bash
cd Wireless_Project/implementation/SecFedPROM
python main.py
```

---

# 🎥 Project Demonstration Video

YouTube Project Link:

[Watch Project Demo](https://youtu.be/NFdbD4v6Y9g)

---


