"""
Q2(ii): Adversarial Detection Model
- Generates adversarial examples using PGD and BIM via IBM ART
- Trains a ResNet-34 binary classifier (clean=0, adversarial=1) for each attack
- Reports detection accuracy (target >= 70%)
- Logs everything to WandB
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms, models
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier

# Config
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VICTIM_W    = "weights/resnet18_cifar10.pth"
BATCH_SIZE  = 256
DET_EPOCHS  = 20
DET_LR      = 1e-3
VAL_SPLIT   = 0.2
SEED        = 42

# Attack hyper-params
PGD_EPS      = 0.03;  PGD_STEP = 0.007; PGD_ITER = 20
BIM_EPS      = 0.03;  BIM_STEP = 0.007; BIM_ITER = 20

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
CLASSES      = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs("weights", exist_ok=True)
os.makedirs("vis",     exist_ok=True)

# WandB 
wandb.init(project="DLOps-Ass5-Q2", name="adversarial_detection_pgd_bim")

# ─── Load victim ResNet-18 ────────────────────────────────────────────────────
def get_resnet18_cifar10():
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, 10)
    return m

victim = get_resnet18_cifar10().to(DEVICE)
victim.load_state_dict(torch.load(VICTIM_W, map_location=DEVICE))
victim.eval()

# ─── CIFAR-10 data ────────────────────────────────────────────────────────────
test_tf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
test_ldr = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

clean_imgs, clean_labels = [], []
for x, y in test_ldr:
    clean_imgs.append(x); clean_labels.append(y)
clean_imgs   = torch.cat(clean_imgs,   0)    # (10000,3,32,32) normalised
clean_labels = torch.cat(clean_labels, 0)
clean_np     = clean_imgs.numpy()

# ART victim classifier 
criterion_victim = nn.CrossEntropyLoss()
art_victim = PyTorchClassifier(
    model       = victim,
    loss        = criterion_victim,
    input_shape = (3, 32, 32),
    nb_classes  = 10,
    clip_values = (
        float(((0 - CIFAR10_MEAN) / CIFAR10_STD).min()),
        float(((1 - CIFAR10_MEAN) / CIFAR10_STD).max()),
    ),
    device_type = "gpu" if torch.cuda.is_available() else "cpu",
)

#  Helper: build binary detection dataset
def make_detection_dataset(clean_tensor, adv_tensor):
    """
    Returns a TensorDataset with:
        X = concat(clean, adv)
        Y = concat(0s, 1s)
    """
    X = torch.cat([clean_tensor, adv_tensor], dim=0)
    Y = torch.cat([torch.zeros(len(clean_tensor), dtype=torch.long),
                   torch.ones( len(adv_tensor),   dtype=torch.long)], dim=0)
    idx = torch.randperm(len(X))
    return TensorDataset(X[idx], Y[idx])

# Detector model: ResNet-34 binary 
def get_detector():
    m = models.resnet34(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, 2)           # binary: clean vs adversarial
    return m.to(DEVICE)

# Train / Eval helpers
def train_detector(detector, train_ldr, val_ldr, attack_name, epochs=DET_EPOCHS):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(detector.parameters(), lr=DET_LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val   = 0.0
    save_path  = f"weights/detector_{attack_name}.pth"

    for epoch in range(1, epochs + 1):
        # ── train ──
        detector.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = detector(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * x.size(0)
            tr_correct += out.argmax(1).eq(y).sum().item()
            tr_total   += y.size(0)
        tr_loss /= tr_total
        tr_acc   = 100.0 * tr_correct / tr_total

        # val
        detector.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_ldr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out  = detector(x)
                loss = criterion(out, y)
                val_loss    += loss.item() * x.size(0)
                val_correct += out.argmax(1).eq(y).sum().item()
                val_total   += y.size(0)
        val_loss /= val_total
        val_acc   = 100.0 * val_correct / val_total
        scheduler.step()

        wandb.log({f"{attack_name}/epoch":    epoch,
                   f"{attack_name}/tr_loss":  tr_loss,
                   f"{attack_name}/tr_acc":   tr_acc,
                   f"{attack_name}/val_loss": val_loss,
                   f"{attack_name}/val_acc":  val_acc})
        print(f"[{attack_name}] Epoch {epoch:2d}/{epochs}  "
              f"TrainAcc {tr_acc:.2f}%  ValAcc {val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(detector.state_dict(), save_path)

    print(f"[{attack_name}] Best Val Accuracy: {best_val:.2f}%")
    return best_val, save_path


def test_detector(detector, test_ldr):
    detector.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += detector(x).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

# Denorm helper for WandB images 
def denorm(t):
    t = t.numpy()
    t = t * CIFAR10_STD[:, None, None] + CIFAR10_MEAN[:, None, None]
    return np.clip(t.transpose(1, 2, 0), 0, 1)



print("\n" + "="*60)
print("Generating PGD adversarial examples …")
pgd_attack  = ProjectedGradientDescent(
    estimator  = art_victim,
    eps        = PGD_EPS,
    eps_step   = PGD_STEP,
    max_iter   = PGD_ITER,
    batch_size = BATCH_SIZE,
)
adv_pgd_np  = pgd_attack.generate(x=clean_np)
adv_pgd     = torch.tensor(adv_pgd_np)

# Log 10 WandB samples (clean + PGD)
wandb_pgd = []
for i in range(10):
    wandb_pgd.append(wandb.Image(denorm(clean_imgs[i]),
                                 caption=f"Clean  {CLASSES[clean_labels[i]]}"))
    wandb_pgd.append(wandb.Image(denorm(adv_pgd[i]),
                                 caption="PGD Adversarial"))
wandb.log({"pgd_samples": wandb_pgd})

# Build dataset, split train/val/test
pgd_ds   = make_detection_dataset(clean_imgs, adv_pgd)
n        = len(pgd_ds)
n_val    = int(n * VAL_SPLIT * 0.5)
n_test   = int(n * VAL_SPLIT * 0.5)
n_train  = n - n_val - n_test
tr_ds, val_ds, te_ds = random_split(pgd_ds, [n_train, n_val, n_test],
                                     generator=torch.Generator().manual_seed(SEED))
tr_ldr   = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_ldr  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
te_ldr   = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

detector_pgd         = get_detector()
best_pgd, pgd_w_path = train_detector(detector_pgd, tr_ldr, val_ldr, "PGD")
detector_pgd.load_state_dict(torch.load(pgd_w_path, map_location=DEVICE))
pgd_test_acc = test_detector(detector_pgd, te_ldr)
print(f"[PGD] Final Test Detection Accuracy: {pgd_test_acc:.2f}%")
wandb.log({"PGD/test_detection_acc": pgd_test_acc})


# (b)  BIM Attack  →  Detector-B

print("\n" + "="*60)
print("Generating BIM adversarial examples …")
bim_attack  = BasicIterativeMethod(
    estimator  = art_victim,
    eps        = BIM_EPS,
    eps_step   = BIM_STEP,
    max_iter   = BIM_ITER,
    batch_size = BATCH_SIZE,
)
adv_bim_np  = bim_attack.generate(x=clean_np)
adv_bim     = torch.tensor(adv_bim_np)

# Log 10 WandB samples (clean + BIM)
wandb_bim = []
for i in range(10):
    wandb_bim.append(wandb.Image(denorm(clean_imgs[i]),
                                 caption=f"Clean  {CLASSES[clean_labels[i]]}"))
    wandb_bim.append(wandb.Image(denorm(adv_bim[i]),
                                 caption="BIM Adversarial"))
wandb.log({"bim_samples": wandb_bim})

bim_ds   = make_detection_dataset(clean_imgs, adv_bim)
n        = len(bim_ds)
n_val    = int(n * VAL_SPLIT * 0.5)
n_test   = int(n * VAL_SPLIT * 0.5)
n_train  = n - n_val - n_test
tr_ds, val_ds, te_ds = random_split(bim_ds, [n_train, n_val, n_test],
                                     generator=torch.Generator().manual_seed(SEED))
tr_ldr   = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_ldr  = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
te_ldr   = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

detector_bim         = get_detector()
best_bim, bim_w_path = train_detector(detector_bim, tr_ldr, val_ldr, "BIM")
detector_bim.load_state_dict(torch.load(bim_w_path, map_location=DEVICE))
bim_test_acc = test_detector(detector_bim, te_ldr)
print(f"[BIM] Final Test Detection Accuracy: {bim_test_acc:.2f}%")
wandb.log({"BIM/test_detection_acc": bim_test_acc})

# Comparison table 
cmp_table = wandb.Table(columns=["Attack","Test Detection Accuracy (%)","Target"])
cmp_table.add_data("PGD", round(pgd_test_acc, 2), "≥70%")
cmp_table.add_data("BIM", round(bim_test_acc, 2), "≥70%")
wandb.log({"detection_comparison": cmp_table})

# ─── Bar chart ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["PGD", "BIM"], [pgd_test_acc, bim_test_acc], color=["steelblue","salmon"],
       width=0.4, edgecolor="black")
ax.axhline(70, ls="--", color="red", label="70% target")
ax.set_ylim(0, 100)
ax.set_ylabel("Detection Accuracy (%)")
ax.set_title("Adversarial Detection: PGD vs BIM")
ax.legend(); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("vis/detection_comparison.png", dpi=150)
wandb.log({"detection_bar_chart": wandb.Image("vis/detection_comparison.png")})
print("Saved vis/detection_comparison.png")

#  Summary print 
print("\n" + "="*60)
print("SUMMARY")
print(f"  PGD Detection Accuracy : {pgd_test_acc:.2f}%  {'✔' if pgd_test_acc>=70 else '✘'}")
print(f"  BIM Detection Accuracy : {bim_test_acc:.2f}%  {'✔' if bim_test_acc>=70 else '✘'}")
print("="*60)

wandb.finish()