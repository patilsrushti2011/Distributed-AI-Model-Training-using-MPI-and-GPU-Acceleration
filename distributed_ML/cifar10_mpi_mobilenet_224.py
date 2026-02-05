# cifar10_mpi_mobilenet_224.py
# CIFAR-10 + MobileNetV2 + MPI + DDP, 224x224, same augments as serial version

import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

from mpi4py import MPI
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
from torchvision import models

# ---------------- MPI + Distributed init ----------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

def setup_distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cpu")

    return device, local_rank, backend

def cleanup():
    dist.destroy_process_group()

# ---------------- Main ----------------

def main():
    if rank == 0:
        print("=" * 70)
        print("MPI + DDP MobileNetV2 CIFAR-10 (224x224)")
        print("=" * 70)

    torch.manual_seed(42)
    device, local_rank, backend = setup_distributed()

    if rank == 0:
        print(f"Backend: {backend}, World size: {world_size}, Device(rank0): {device}")

    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # --------- transforms: 224x224 + strong augments ----------

    IMG_SIZE = 224

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --------- CIFAR-10 datasets (rank 0 downloads) ---------

    if rank == 0:
        print("Downloading CIFAR-10 (if not present)...")
        _ = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        _ = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )

    dist.barrier()

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_transform
    )

    if rank == 0:
        print("Train samples:", len(train_dataset))
        print("Test samples:", len(test_dataset))

    # --------- Distributed samplers + loaders ---------------

    batch_size = 128

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=test_sampler, num_workers=2
    )

    # ---------------- MobileNetV2 model ----------------

    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 10)
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if rank == 0:
        total_params = sum(p.numel() for p in ddp_model.parameters())
        print(f"Total parameters: {total_params}")
        print("Starting distributed training...\n")

    # ---------------- Training loop ----------------

    EPOCHS = 20
    best_acc = 0.0
    best_state = deepcopy(ddp_model.module.state_dict())
    total_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        # ---- train ----
        ddp_model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * imgs.size(0)
            running_correct += torch.sum(preds == labels).item()
            running_total += labels.size(0)

        # aggregate train loss across ranks
        train_loss_tensor = torch.tensor(
            [running_loss, running_total],
            device=device, dtype=torch.float64
        )
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        global_loss = train_loss_tensor[0].item()
        global_total = train_loss_tensor[1].item()
        epoch_loss = global_loss / global_total
        epoch_acc = running_correct / running_total  # local approx

        # ---- evaluate ----
        ddp_model.eval()
        test_correct = 0
        test_total = 0
        test_loss_sum = 0.0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = ddp_model(imgs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                test_loss_sum += loss.item() * imgs.size(0)
                test_correct += torch.sum(preds == labels).item()
                test_total += labels.size(0)

        # aggregate test loss across ranks
        test_loss_tensor = torch.tensor(
            [test_loss_sum, test_total],
            device=device, dtype=torch.float64
        )
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        global_test_loss = test_loss_tensor[0].item()
        global_test_total = test_loss_tensor[1].item()
        test_loss = global_test_loss / global_test_total
        test_acc = test_correct / test_total  # local approx

        scheduler.step()
        epoch_time = time.time() - epoch_start

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Time: {epoch_time:.2f}s "
                f"Train Loss: {epoch_loss:.4f} "
                f"Test Loss: {test_loss:.4f} "
                f"Test Acc(local): {test_acc:.4f}"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(ddp_model.module.state_dict())

    total_time = time.time() - total_start
    if rank == 0:
        print(f"\nBest local test accuracy: {best_acc:.4f}")
        print(
            f"Total training time: {total_time:.2f}s "
            f"({total_time/60:.2f} min)"
        )
        torch.save(best_state, "best_mobilenetv2_cifar10_224_mpi.pth")
        print("Saved best_mobilenetv2_cifar10_224_mpi.pth")

    cleanup()

if __name__ == "__main__":
    main()

