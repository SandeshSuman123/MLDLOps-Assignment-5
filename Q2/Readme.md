# Q2 – Adversarial Attacks with IBM ART

## Overview

| Script | Purpose |
|---|---|
| `train_resnet18.py` | Train ResNet-18 on CIFAR-10 from scratch |
| `fgsm_attack.py` | FGSM: from-scratch implementation vs IBM ART |
| `adversarial_detection.py` | ResNet-34 binary detectors for PGD and BIM attacks |

---

## Environment Setup

```bash
source .venv/bin/activate
pip install torch torchvision wandb numpy matplotlib adversarial-robustness-toolbox huggingface_hub
```

---

## Running the Code

```bash
# Step 1 - Train ResNet-18
python train_resnet18.py

# Step 2 - FGSM Attack
python fgsm_attack.py

# Step 3 - Adversarial Detection
python adversarial_detection.py
```

---

## Results

### ResNet-18 Training on CIFAR-10

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1  | 5.2320 | 10.24% | 2.3023 | 10.43% |
| 5  | 2.2075 | 14.94% | 2.1109 | 18.28% |
| 10 | 1.6779 | 34.86% | 1.7809 | 31.34% |
| 15 | 1.0889 | 60.58% | 1.1153 | 60.29% |
| 20 | 0.6440 | 77.24% | 0.7131 | 75.47% |
| 25 | 0.4187 | 85.52% | 0.5132 | 82.44% |
| 30 | 0.3146 | 89.30% | 0.4329 | 85.46% |

**Best Test Accuracy: 85.46%** ✔ (Target ≥ 72%)

---

### Q2(i): FGSM Comparison — Clean vs Adversarial Accuracy

Clean Accuracy: **85.49%**

| ε | FGSM Scratch Acc | FGSM ART Acc | Scratch Drop | ART Drop |
|---|---|---|---|---|
| 0.01 | 66.50% | 73.75% | 18.99% | 11.74% |
| 0.02 | 48.25% | 56.55% | 37.24% | 28.94% |
| 0.03 | 34.35% | 42.88% | 51.14% | 42.61% |
| 0.05 | 18.21% | 26.59% | 67.28% | 58.90% |
| 0.10 |  6.18% | 14.04% | 79.31% | 71.45% |

**Analysis:**
- Both attacks degrade accuracy significantly as ε increases.
- FGSM from scratch is slightly stronger (lower accuracy) than IBM ART at every ε, likely due to differences in clipping behavior.
- At ε=0.10, scratch FGSM reduces accuracy from 85.49% to just 6.18% — near random guessing for 10 classes.

---

### Q2(ii): Adversarial Detection Results

| Attack | Detection Accuracy | Target |
|---|---|---|
| PGD | 51.80% | ≥ 70% |
| BIM | 48.55% | ≥ 70% |

**Note:** Detection accuracy is near random (50%) because PGD and BIM with small ε (0.03) produce adversarial examples that are visually and statistically very close to clean images, making binary detection extremely hard for ResNet-34 without adversarial training. To improve above 70%, stronger attack parameters (larger ε) or feature-space detectors are needed.

---

## WandB & HuggingFace Links

- **WandB Project:** https://wandb.ai/sandeshsuman2000-iit-jodhpur/DLOps-Ass5-Q2
- **HuggingFace Model:** https://huggingface.co/sandesh2233/dlops-ass5-q2

---

## Attack Hyper-parameters

| Attack | ε | Step size | Iterations |
|---|---|---|---|
| FGSM | 0.01–0.10 | — | 1 |
| PGD  | 0.03 | 0.007 | 20 |
| BIM  | 0.03 | 0.007 | 20 |
