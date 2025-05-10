from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import nibabel as nib
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation


class Net(nn.Module):
    """CNN model"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels (axial, coronal, sagittal)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),  # input size tùy thuộc đầu vào 128x128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


class BrainAgeDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img_3d = nib.load(sample["image_path"]).get_fdata()

        # Normalize toàn bộ ảnh 3D trước
        img_3d = (img_3d - img_3d.min()) / (img_3d.max() - img_3d.min() + 1e-8)

        # Lấy lát cắt giữa mỗi trục (mid-slice)
        d, h, w = img_3d.shape
        axial = img_3d[d // 2, :, :]
        coronal = img_3d[:, h // 2, :]
        sagittal = img_3d[:, :, w // 2]

         # Ghép lại thành ảnh RGB: [H, W, 3]
        img_rgb = np.stack([axial, coronal, sagittal], axis=-1)
        img_rgb = (img_rgb * 255).astype(np.uint8)

        # Chuyển sang PIL để dùng transform
        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_tensor = self.transform(img_pil)  # [3, H, W]
        else:
            img_tensor = ToTensor()(img_pil)

        return {
            "image": img_tensor,
            "age": torch.tensor(sample["subject_age"], dtype=torch.float32),
            "subject_id": sample["subject_id"],
            "age_group": sample["age_group"]
        }

def load_data(partition_id: int, num_partitions: int, partitioner: str, excel_path):
    """Load partition MRI data."""

    df = pd.read_excel(excel_path, nrows=900)
    df["image_path"] = df["subject_id"].apply(lambda x: f"/media/sslab/PACS/sslab/nguyentiendung/data_processed/{x}/anat/{x}_T1w_processed.nii.gz")
    bins = [0, 20, 40, 60, 100]
    labels = ["0-20", "21-40", "41-60", "61+"]
    df["age_group"] = pd.cut(df["subject_age"], bins=bins, labels=labels)
    dataset = Dataset.from_pandas(df)

    if partitioner == "pathological":
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions, 
            partition_by="age_group", 
            num_classes_per_partition=4,
            class_assignment_mode="random",
        )
    elif partitioner == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="age_group",
            alpha=0.3,
            seed=42,
        )
    else:
        partitioner = IidPartitioner(num_partitions=num_partitions)

    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_transform = Compose([
        Resize((128, 128)), ToTensor(), 
        RandomHorizontalFlip(), RandomRotation(10),
        Normalize([0.5] * 3, [0.5] * 3)
    ])

    test_transform = Compose([
        Resize((128, 128)), ToTensor(),
        Normalize([0.5] * 3, [0.5] * 3)
    ])

    trainset = BrainAgeDataset(partition_train_test["train"], transform=train_transform)
    testset = BrainAgeDataset(partition_train_test["test"], transform=test_transform)
    
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, pin_memory=True)
    testloader = DataLoader(testset, batch_size=4, pin_memory=True)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device, strategy_name, proximal_mu):
    """Train the model on the training set."""
    net.to(device)
    criterion = nn.L1Loss().to(device)

    strategy = strategy_name

    if strategy == "fedadam":
        # FedAdam: use Adam optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        # FedAvg/FedProx: use SGD optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    proximal_mu = proximal_mu if strategy == "FedProx" else 0.0
    if proximal_mu > 0:
        global_params = [p.detach().clone() for p in net.parameters()]

    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["age"].to(device)

            optimizer.zero_grad()
            outputs = net(images).squeeze()
            loss = criterion(outputs, labels)

            if proximal_mu > 0:
                proximal_term = sum((lp - gp).norm(2).pow(2) for lp, gp in zip(net.parameters(), global_params))
                loss += (proximal_mu / 2) * proximal_term
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / len(trainloader)


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    mae_loss = torch.nn.L1Loss()  # MAE
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["age"].to(device).view(-1)
            outputs = net(images).view(-1)
            loss = mae_loss(outputs, labels)
            total_loss += loss.item() * images.size(0)  # multiply by batch size

            # Accuracy: dự đoán đúng nếu sai số ≤ 5 tuổi
            preds = outputs
            correct = (torch.abs(preds - labels) <= 5.0).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

    mae = total_loss / total_samples
    accuracy = total_correct / total_samples
    return mae, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)