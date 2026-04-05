import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import wandb
import numpy as np
from torch.cuda.amp import GradScaler, autocast

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 30
BATCH_SIZE  = 1024
NUM_WORKERS = 16
LR          = 0.4
MOMENTUM    = 0.9
WD          = 5e-4
SAVE_PATH   = "weights/resnet18_cifar10.pth"
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

wandb.init(project="DLOps-Ass5-Q2", name="resnet18_train_cifar10")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_dataset = datasets.CIFAR10(root="data", train=True,  download=True, transform=train_transform)
test_dataset  = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def get_resnet18_cifar10():
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model

model = get_resnet18_cifar10().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler    = GradScaler()

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total

os.makedirs("weights", exist_ok=True)
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
    te_loss, te_acc = eval_epoch(model, test_loader, criterion)
    scheduler.step()

    wandb.log({"epoch": epoch,
               "train/loss": tr_loss, "train/acc": tr_acc,
               "test/loss":  te_loss, "test/acc":  te_acc})

    print(f"Epoch {epoch:3d}/{EPOCHS}  Train {tr_loss:.4f}/{tr_acc:.2f}%  Test {te_loss:.4f}/{te_acc:.2f}%")

    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  Saved best model (test acc = {best_acc:.2f}%)")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")
wandb.summary["best_test_accuracy"] = best_acc
wandb.finish()
