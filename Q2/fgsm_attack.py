
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS    = "weights/resnet18_cifar10.pth"
EPSILONS   = [0.01, 0.02, 0.03, 0.05, 0.1]
BATCH_SIZE = 256
NUM_SAMPLES_WANDB = 10
CLASSES    = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

wandb.init(project="DLOps-Ass5-Q2", name="fgsm_scratch_vs_art")

def get_resnet18_cifar10():
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model

model = get_resnet18_cifar10().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.eval()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

all_imgs, all_labels = [], []
for imgs, labels in test_loader:
    all_imgs.append(imgs)
    all_labels.append(labels)
all_imgs   = torch.cat(all_imgs,   dim=0)
all_labels = torch.cat(all_labels, dim=0)
all_imgs_np = all_imgs.numpy()

def accuracy(model, imgs_tensor, labels_tensor, batch=256):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs_tensor), batch):
            x = imgs_tensor[i:i+batch].to(DEVICE)
            y = labels_tensor[i:i+batch].to(DEVICE)
            correct += model(x).argmax(1).eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

clean_acc = accuracy(model, all_imgs, all_labels)
print(f"Clean accuracy: {clean_acc:.2f}%")
wandb.log({"clean_accuracy": clean_acc})

criterion = nn.CrossEntropyLoss()

def fgsm_scratch(imgs, labels, eps):
    # use a fresh leaf tensor so .grad is always populated
    x = imgs.clone().to(DEVICE)
    x_adv = x.detach().requires_grad_(True)
    loss = criterion(model(x_adv), labels.to(DEVICE))
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        perturbation = eps * x_adv.grad.sign()
        adv = x_adv + perturbation
    return adv.detach().cpu()

scratch_accs = {}
adv_scratch_all = None

for eps in EPSILONS:
    adv_chunks = []
    for i in range(0, len(all_imgs), BATCH_SIZE):
        x = all_imgs[i:i+BATCH_SIZE]
        y = all_labels[i:i+BATCH_SIZE]
        adv_chunks.append(fgsm_scratch(x, y, eps))
    adv_tensor = torch.cat(adv_chunks, dim=0)
    if eps == EPSILONS[-1]:
        adv_scratch_all = adv_tensor
    acc = accuracy(model, adv_tensor, all_labels)
    scratch_accs[eps] = acc
    print(f"[FGSM Scratch] eps={eps:.3f}  acc={acc:.2f}%")
    wandb.log({f"fgsm_scratch/acc_eps_{eps}": acc, "epsilon": eps})

art_classifier = PyTorchClassifier(
    model       = model,
    loss        = criterion,
    input_shape = (3, 32, 32),
    nb_classes  = 10,
    clip_values = (
        float(((0 - CIFAR10_MEAN) / CIFAR10_STD).min()),
        float(((1 - CIFAR10_MEAN) / CIFAR10_STD).max()),
    ),
    device_type = "gpu" if torch.cuda.is_available() else "cpu",
)

art_accs    = {}
adv_art_all = None

for eps in EPSILONS:
    attack = FastGradientMethod(estimator=art_classifier, eps=eps, batch_size=BATCH_SIZE)
    adv_np = attack.generate(x=all_imgs_np)
    adv_t  = torch.tensor(adv_np)
    if eps == EPSILONS[-1]:
        adv_art_all = adv_t
    acc = accuracy(model, adv_t, all_labels)
    art_accs[eps] = acc
    print(f"[FGSM ART]    eps={eps:.3f}  acc={acc:.2f}%")
    wandb.log({f"fgsm_art/acc_eps_{eps}": acc, "epsilon": eps})

table = wandb.Table(columns=["epsilon","clean_acc","scratch_acc","art_acc","scratch_drop","art_drop"])
for eps in EPSILONS:
    table.add_data(eps, clean_acc, scratch_accs[eps], art_accs[eps],
                   clean_acc - scratch_accs[eps], clean_acc - art_accs[eps])
wandb.log({"fgsm_comparison_table": table})

def denorm(tensor):
    t = tensor.numpy()
    t = t * CIFAR10_STD[:, None, None] + CIFAR10_MEAN[:, None, None]
    return np.clip(t.transpose(1, 2, 0), 0, 1)

os.makedirs("vis", exist_ok=True)
fig, axes = plt.subplots(NUM_SAMPLES_WANDB, 3, figsize=(9, 3 * NUM_SAMPLES_WANDB))
wandb_images = []
for idx in range(NUM_SAMPLES_WANDB):
    orig_img    = denorm(all_imgs[idx])
    scratch_img = denorm(adv_scratch_all[idx])
    art_img     = denorm(adv_art_all[idx])
    lbl         = CLASSES[all_labels[idx].item()]
    axes[idx,0].imshow(orig_img);    axes[idx,0].set_title(f"Original ({lbl})")
    axes[idx,1].imshow(scratch_img); axes[idx,1].set_title("FGSM Scratch")
    axes[idx,2].imshow(art_img);     axes[idx,2].set_title("FGSM ART")
    for ax in axes[idx]: ax.axis("off")
    wandb_images.append(wandb.Image(orig_img,    caption=f"Original label={lbl}"))
    wandb_images.append(wandb.Image(scratch_img, caption=f"FGSM Scratch eps={EPSILONS[-1]}"))
    wandb_images.append(wandb.Image(art_img,     caption=f"FGSM ART eps={EPSILONS[-1]}"))

plt.tight_layout()
plt.savefig("vis/fgsm_comparison.png", dpi=150)
wandb.log({"fgsm_sample_images": wandb_images})
wandb.log({"fgsm_comparison_grid": wandb.Image("vis/fgsm_comparison.png")})

fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot(EPSILONS, [scratch_accs[e] for e in EPSILONS], "o-", label="FGSM Scratch")
ax.plot(EPSILONS, [art_accs[e]    for e in EPSILONS], "s--",label="FGSM ART")
ax.axhline(clean_acc, ls=":", color="gray", label=f"Clean ({clean_acc:.1f}%)")
ax.set_xlabel("Epsilon"); ax.set_ylabel("Test Accuracy (%)")
ax.set_title("FGSM: Perturbation Strength vs Accuracy Drop")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig("vis/fgsm_perf_drop.png", dpi=150)
wandb.log({"fgsm_perf_drop_plot": wandb.Image("vis/fgsm_perf_drop.png")})

print("Done!")
wandb.finish()