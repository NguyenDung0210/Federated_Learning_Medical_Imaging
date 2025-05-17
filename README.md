# Research and Development of a Federated Learning System for MRI Data

This project develops and evaluates a **Federated Learning (FL)** system using both benchmark and medical imaging data. The FL system is built with **Flower**, supports real-time logging via **Flask + SocketIO**, and is simulated on a single machine.

## 📌 Project Summary

- **Stage 1: CIFAR-10 Benchmark**
  - Test FL system with various data partitioning methods and aggregation strategies.
  - Evaluate flexibility and performance in different federated setups.

- **Stage 2: Brain Age Prediction**
  - Apply the system to a real-world medical task using 3D MRI data.
  - Predict the brain age from MRI slices using federated training.

## 🔑 Key Features

- Support for multiple FL strategies: `FedAvg`, `FedProx`, `FedAdam`
- Partitioning options: `iid`, `shard`, `pathological`, `dirichlet`
- Integration of real-time logging using Flask and Socket.IO
- Modular structure for easy switching between CIFAR-10 and MRI
- GPU support for client training

## 🧠 Dataset Description

The brain imaging dataset used in this project includes:

- **Training set**: 1,000 T1-weighted 3D MRI scans  
- **Test set**: 500 T1-weighted 3D MRI scans  
- **Resolution**: 130×130×130 voxels  
- **Preprocessing**: No skull-stripping; both brain tissue and skull are visible  
- **Usage**: Internal research only; not publicly available  

This dataset is used to simulate a federated learning environment, where data is split across multiple clients.

## 📁 Project Structure
```
.
├── fl-brain/
│ ├── brain_age_prediction.ipynb
│ ├── fl_brain/
│ │ ├── init.py
│ │ ├── client_app.py
│ │ ├── my_strategy.py
│ │ ├── server_app.py
│ │ ├── socket_emit.py
│ │ └── task.py
│ ├── global_model_final_fedprox_dirichlet.pt
│ ├── pyproject.toml
│ ├── results_fedprox_dirichlet.json
│ ├── test.ipynb
│ └── visualize.ipynb
├── fl-cifar10/
│ ├── evaluate.ipynb
│ ├── fl_cifar10/
│ │ ├── init.py
│ │ ├── client_app.py
│ │ ├── my_strategy.py
│ │ ├── server_app.py
│ │ ├── socket_emit.py
│ │ └── task.py
│ ├── partitioner_visualized.ipynb
│ ├── pyproject.toml
│ ├── results/
│ └── visualize_result.ipynb
└── web/
│ ├── app.py
│ ├── static/
│ │ └── style.css
│ ├── templates/
│ │ └── index.html
├── requirements.txt
├── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Federated Learning System
➤ CIFAR-10 Experiments
```bash
cd fl-cifar10
```

Edit pyproject.toml and set:

- num-server-rounds
- strategy = "fedavg" / "fedprox" / "fedadam"
- partitioner = "iid" / "shard" / "pathological" / "dirichlet"
- fraction-fit, local-epochs, options.num-supernodes

Then run:
``` bash
flwr run .
```

➤ Brain MRI Experiments
```bash
cd fl-brain
```

Edit pyproject.toml and set:

- num-server-rounds
- strategy = "fedavg" / "fedprox" / "fedadam"
- partitioner = "iid" / "shard" / "pathological" / "dirichlet"
- fraction-fit, local-epochs, options.num-supernodes

Then run:
``` bash
flwr run .
```

### 4. Run with GPU
Make sure PyTorch with CUDA is installed. I`n pyproject.toml, set:
```toml
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.25
```

### 5. Launch Web Interface (optional)
```bash
cd web
python app.py
```
Open your browser at: http://localhost:5000