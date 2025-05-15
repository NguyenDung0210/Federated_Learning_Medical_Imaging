# Federated Learning System with Flower & PyTorch

A two-stage federated learning (FL) system for:
1. **CIFAR-10 classification** (simulated environment)
2. **Brain MRI age prediction** (medical imaging application)

## System Overview
- **Framework**: Flower (flwr) + PyTorch
- **Execution**: Local simulation on a single machine
- **Configuration**: Centralized control via `pyproject.toml`

### Stage 1: CIFAR-10 Experiment
- Simulates 10 virtual clients with partitioned CIFAR-10 data
- Includes:
  - Custom FL strategies (`my_strategy.py`)
  - Dataset handling (`task.py`)
  - Server/client logic (`server_app.py`, `client_app.py`)

### Stage 2: Brain MRI Application
- Simulates 3 virtual clients with brain MRI data
- Reuses Stage 1 infrastructure with modified:
  - Data loading (MRI-specific preprocessing)
  - Model output (regression for age prediction)

## Project Structure
.
├── fl-cifar10/ # Stage 1: CIFAR-10 implementation
│ ├── server_app.py
│ ├── client_app.py
│ ├── task.py
│ ├── my_strategy.py
│ └── pyproject.toml
├── fl-brain/ # Stage 2: MRI implementation
│ ├── server_app.py
│ ├── client_app.py
│ ├── task.py
│ ├── my_strategy.py
│ └── pyproject.toml

## Quick Start
### For CIFAR-10 (Stage 1)
```bash
cd fl-cifar10
pip install -e .  # Install dependencies
flwr run .       # Launch simulation

### For Brain MRI (Stage 2)
```bash
cd fl-brain
pip install -e .  # Install dependencies
flwr run .       # Launch simulation

## Configuration
Modify pyproject.toml to control experiments:
strategy = "fedavg"       # "fedavg"/"fedprox"/"fedadam"
partitioner = "dirichlet" # "iid"/"shard"/"pathological"/"dirichlet"
fraction-fit = 1
local-epochs = 10

## Key Features
✅ Single-machine FL simulation

✅ Reusable infrastructure across stages

✅ Configuration-driven experiments

✅ Custom strategies support