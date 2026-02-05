# cifar10_serial_mobilenet_224.py
# CIFAR-10 training with MobileNetV2 (stronger augmentations, 224x224, top-3 inference)

import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F

# ---------------- Device ----------------

device = torch.device("cpu")
print("Device:", device)

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------- Transforms ----------------

IMG_SIZE = 224  # increased from 128 for more detail

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

# ---------------- CIFAR-10 data ----------------

print("Downloading CIFAR-10 (if needed)...")
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# ---------------- MobileNetV2 model ----------------

model = models.mobilenet_v2(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)
print("Starting serial training...\n")

# ---------------- Training loop ----------------

EPOCHS = 20
best_acc = 0.0
best_state = deepcopy(model.state_dict())
total_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # ---- train ----
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += torch.sum(preds == labels).item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)

    # ---- evaluate ----
    model.eval()
    test_loss_sum = 0.0
    test_correct = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss_sum += loss.item() * imgs.size(0)
            test_correct += torch.sum(preds == labels).item()

    test_loss = test_loss_sum / len(test_dataset)
    test_acc = test_correct / len(test_dataset)

    scheduler.step()
    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Time: {epoch_time:.2f}s "
        f"Train Loss: {epoch_loss:.4f} "
        f"Train Acc: {epoch_acc:.4f} "
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_acc:.4f}"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        best_state = deepcopy(model.state_dict())

total_time = time.time() - total_start
print(f"\nBest test accuracy: {best_acc:.4f}")
print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")

model.load_state_dict(best_state)
torch.save(model.state_dict(), "best_mobilenetv2_cifar10_224.pth")
print("Saved best_mobilenetv2_cifar10_224.pth")

# ---------------- Inference helper ----------------

inference_transform = test_transform  # same as test

def predict_cifar10_image(img_path, topk=3, conf_threshold=0.5):
    """
    Returns top-k (label, confidence) and also prints them.
    If best confidence < conf_threshold, treat as 'uncertain'.
    """
    img = Image.open(img_path).convert("RGB")
    x = inference_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)[0]

        topk_conf, topk_idx = torch.topk(probs, k=topk)
        topk_labels = [classes[i.item()] for i in topk_idx]
        topk_conf = [float(c) for c in topk_conf]

    print("Top-{} predictions:".format(topk))
    for label, c in zip(topk_labels, topk_conf):
        print(f"  {label}: {c:.3f}")

    best_label = topk_labels[0]
    best_conf = topk_conf[0]

    if best_conf < conf_threshold:
        print(f"Prediction uncertain (best_conf={best_conf:.3f} < {conf_threshold})")
    else:
        print(f"Predicted: {best_label} confidence: {best_conf:.3f}")

    return list(zip(topk_labels, topk_conf))

# example (uncomment and change path):
# top3 = predict_cifar10_image("/home/chuk398/image_classification/cat.jpg", topk=3, conf_threshold=0.5)

