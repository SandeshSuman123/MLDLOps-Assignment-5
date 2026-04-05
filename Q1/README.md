# Assignment 5 — DLOps: Sandesh Suman(M25CSA034)

> **Branch:** `Assignment-5`
> **WandB Project:** [DLOps-Ass5-Q1-ViT-LoRA](https://wandb.ai/sandeshsuman2000-iit-jodhpur/m25csa034_MLDLops)
> **HuggingFace Model:** [sandesh2233/vit-lora-cifar100](https://huggingface.co/sandesh2233/vit-lora-cifar100)

---

## Project Structure

```
Assignment-5/
├── Dockerfile
├── requirements.txt
├── config.py                  # All hyperparameters
├── dataset.py                 # CIFAR-100 data loading
├── model.py                   # ViT-S baseline + LoRA builder
├── trainer.py                 # Training loop (AMP + WandB)
├── evaluate.py                # Test accuracy + class-wise histogram
├── train_baseline.py          # Q1.1 — head-only finetuning
├── train_lora.py              # Q1.2 — all 9 LoRA combinations
├── optuna_search.py           # Q1.5 — Optuna HPO
├── generate_report_tables.py  # Print tables + save plots
└── outputs/                   # Weights, JSONs, plots
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/sandesh2233/MLDLOps-Assignment-5.git
cd MLDLOps-Assignment-5
git checkout Assignment-5
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_ENTITY="your_wandb_team"
export HF_TOKEN="your_huggingface_token"
export HF_REPO_ID="your_hf_repo_id"
```

### 5. Docker setup (alternative)
```bash
docker build -t dlops-ass5 .
docker run --gpus all --shm-size=24g -it --rm \
  -v $(pwd):/workspace \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_ENTITY=$WANDB_ENTITY \
  dlops-ass5 bash
```

---

## Running the Code

### Q1.1 — Baseline (head-only, no LoRA)
```bash
python train_baseline.py
```

### Q1.2 — All 9 LoRA combinations
```bash
python train_lora.py
```

To run a single combination:
```bash
python train_lora.py --rank 4 --alpha 8
```

### Q1.5 — Optuna hyperparameter search
```bash
python optuna_search.py
```

### Generate report tables and plots
```bash
python generate_report_tables.py
```

---

## Q1 Results

### Test Accuracy Comparison Table

| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|------|------|-------|---------|--------------|-----------------|
| No   | -    | -     | -       | 79.89%       | 38,500          |
| Yes  | 2    | 2     | 0.1     | 89.07%       | 75,364          |
| Yes  | 2    | 4     | 0.1     | 88.63%       | 75,364          |
| Yes  | 2    | 8     | 0.1     | 89.03%       | 75,364          |
| Yes  | 4    | 2     | 0.1     | 88.91%       | 112,228         |
| Yes  | 4    | 4     | 0.1     | 89.12%       | 112,228         |
| Yes  | 4    | 8     | 0.1     | **89.29%**   | 112,228         |
| Yes  | 8    | 2     | 0.1     | 88.91%       | 185,956         |
| Yes  | 8    | 4     | 0.1     | 89.04%       | 185,956         |
| Yes  | 8    | 8     | 0.1     | 89.21%       | 185,956         |

**Best LoRA config: Rank=4, Alpha=8, Dropout=0.1 — Test Accuracy: 89.29%**
**Optuna best config test accuracy: 88.01%**

---

### Baseline — Epoch-wise Results (No LoRA, head only)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.9274    | 1.4987  | 0.4076    | 0.6952  |
| 2     | 1.1313    | 0.9402  | 0.7398    | 0.7652  |
| 3     | 0.8348    | 0.8138  | 0.7825    | 0.7814  |
| 4     | 0.7325    | 0.7608  | 0.8024    | 0.7897  |
| 5     | 0.6770    | 0.7321  | 0.8143    | 0.7954  |
| 6     | 0.6419    | 0.7153  | 0.8230    | 0.7971  |
| 7     | 0.6196    | 0.7067  | 0.8288    | 0.8013  |
| 8     | 0.6056    | 0.7009  | 0.8326    | 0.8030  |
| 9     | 0.5973    | 0.6984  | 0.8345    | 0.8032  |
| 10    | 0.5935    | 0.6979  | 0.8353    | 0.8034  |

---

### Exp 1 — Rank=2, Alpha=2, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.7263    | 0.9688  | 0.4455    | 0.7899  |
| 2     | 0.6126    | 0.4806  | 0.8423    | 0.8626  |
| 3     | 0.4053    | 0.4163  | 0.8810    | 0.8745  |
| 4     | 0.3444    | 0.3948  | 0.8966    | 0.8777  |
| 5     | 0.3096    | 0.3812  | 0.9070    | 0.8812  |
| 6     | 0.2867    | 0.3743  | 0.9130    | 0.8828  |
| 7     | 0.2712    | 0.3705  | 0.9182    | 0.8840  |
| 8     | 0.2611    | 0.3678  | 0.9201    | 0.8841  |
| 9     | 0.2551    | 0.3670  | 0.9240    | 0.8844  |
| 10    | 0.2522    | 0.3668  | 0.9245    | 0.8847  |

---

### Exp 2 — Rank=2, Alpha=4, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.6293    | 0.8682  | 0.4622    | 0.8025  |
| 2     | 0.5649    | 0.4707  | 0.8503    | 0.8620  |
| 3     | 0.3897    | 0.4120  | 0.8848    | 0.8765  |
| 4     | 0.3312    | 0.3877  | 0.9008    | 0.8800  |
| 5     | 0.2959    | 0.3749  | 0.9105    | 0.8853  |
| 6     | 0.2723    | 0.3680  | 0.9168    | 0.8859  |
| 7     | 0.2556    | 0.3639  | 0.9225    | 0.8866  |
| 8     | 0.2449    | 0.3617  | 0.9263    | 0.8876  |
| 9     | 0.2383    | 0.3606  | 0.9291    | 0.8879  |
| 10    | 0.2351    | 0.3604  | 0.9300    | 0.8877  |

---

### Exp 3 — Rank=2, Alpha=8, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.5350    | 0.7944  | 0.4791    | 0.8118  |
| 2     | 0.5385    | 0.4598  | 0.8549    | 0.8690  |
| 3     | 0.3786    | 0.4095  | 0.8880    | 0.8783  |
| 4     | 0.3202    | 0.3859  | 0.9036    | 0.8818  |
| 5     | 0.2841    | 0.3749  | 0.9133    | 0.8840  |
| 6     | 0.2587    | 0.3676  | 0.9220    | 0.8856  |
| 7     | 0.2413    | 0.3662  | 0.9276    | 0.8872  |
| 8     | 0.2292    | 0.3644  | 0.9316    | 0.8863  |
| 9     | 0.2220    | 0.3638  | 0.9345    | 0.8870  |
| 10    | 0.2182    | 0.3636  | 0.9356    | 0.8873  |

---

### Exp 4 — Rank=4, Alpha=2, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.7292    | 0.9508  | 0.4452    | 0.7952  |
| 2     | 0.6059    | 0.4752  | 0.8441    | 0.8620  |
| 3     | 0.4015    | 0.4139  | 0.8823    | 0.8749  |
| 4     | 0.3403    | 0.3908  | 0.8983    | 0.8801  |
| 5     | 0.3057    | 0.3768  | 0.9081    | 0.8824  |
| 6     | 0.2833    | 0.3712  | 0.9140    | 0.8846  |
| 7     | 0.2674    | 0.3659  | 0.9196    | 0.8871  |
| 8     | 0.2578    | 0.3645  | 0.9228    | 0.8873  |
| 9     | 0.2516    | 0.3635  | 0.9245    | 0.8878  |
| 10    | 0.2487    | 0.3632  | 0.9252    | 0.8878  |

---

### Exp 5 — Rank=4, Alpha=4, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.5862    | 0.8358  | 0.4702    | 0.8084  |
| 2     | 0.5509    | 0.4534  | 0.8505    | 0.8653  |
| 3     | 0.3836    | 0.3997  | 0.8851    | 0.8779  |
| 4     | 0.3255    | 0.3787  | 0.9008    | 0.8839  |
| 5     | 0.2914    | 0.3665  | 0.9108    | 0.8880  |
| 6     | 0.2677    | 0.3610  | 0.9181    | 0.8889  |
| 7     | 0.2513    | 0.3581  | 0.9234    | 0.8889  |
| 8     | 0.2409    | 0.3553  | 0.9274    | 0.8905  |
| 9     | 0.2346    | 0.3543  | 0.9290    | 0.8907  |
| 10    | 0.2312    | 0.3541  | 0.9307    | 0.8905  |

---

### Exp 6 — Rank=4, Alpha=8, Dropout=0.1 (Best)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.5463    | 0.7551  | 0.4744    | 0.8192  |
| 2     | 0.5194    | 0.4493  | 0.8572    | 0.8690  |
| 3     | 0.3692    | 0.3957  | 0.8890    | 0.8809  |
| 4     | 0.3111    | 0.3770  | 0.9050    | 0.8843  |
| 5     | 0.2741    | 0.3682  | 0.9162    | 0.8876  |
| 6     | 0.2490    | 0.3644  | 0.9243    | 0.8888  |
| 7     | 0.2302    | 0.3606  | 0.9307    | 0.8885  |
| 8     | 0.2183    | 0.3596  | 0.9348    | 0.8892  |
| 9     | 0.2115    | 0.3584  | 0.9381    | 0.8889  |
| 10    | 0.2077    | 0.3583  | 0.9393    | 0.8892  |

---

### Exp 7 — Rank=8, Alpha=2, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.7491    | 0.9769  | 0.4422    | 0.7955  |
| 2     | 0.6098    | 0.4711  | 0.8452    | 0.8638  |
| 3     | 0.3990    | 0.4110  | 0.8835    | 0.8750  |
| 4     | 0.3393    | 0.3881  | 0.8982    | 0.8809  |
| 5     | 0.3059    | 0.3756  | 0.9073    | 0.8827  |
| 6     | 0.2833    | 0.3696  | 0.9136    | 0.8855  |
| 7     | 0.2682    | 0.3642  | 0.9187    | 0.8855  |
| 8     | 0.2580    | 0.3625  | 0.9214    | 0.8865  |
| 9     | 0.2521    | 0.3612  | 0.9238    | 0.8867  |
| 10    | 0.2494    | 0.3611  | 0.9246    | 0.8866  |

---

### Exp 8 — Rank=8, Alpha=4, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.5786    | 0.8134  | 0.4736    | 0.8161  |
| 2     | 0.5438    | 0.4467  | 0.8548    | 0.8669  |
| 3     | 0.3766    | 0.3980  | 0.8879    | 0.8770  |
| 4     | 0.3205    | 0.3764  | 0.9030    | 0.8848  |
| 5     | 0.2861    | 0.3664  | 0.9131    | 0.8868  |
| 6     | 0.2632    | 0.3601  | 0.9201    | 0.8885  |
| 7     | 0.2473    | 0.3579  | 0.9260    | 0.8888  |
| 8     | 0.2370    | 0.3553  | 0.9289    | 0.8908  |
| 9     | 0.2306    | 0.3545  | 0.9318    | 0.8909  |
| 10    | 0.2272    | 0.3544  | 0.9327    | 0.8906  |

---

### Exp 9 — Rank=8, Alpha=8, Dropout=0.1

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | 2.5591    | 0.7474  | 0.4809    | 0.8253  |
| 2     | 0.5136    | 0.4397  | 0.8598    | 0.8679  |
| 3     | 0.3616    | 0.3964  | 0.8909    | 0.8785  |
| 4     | 0.3047    | 0.3741  | 0.9070    | 0.8851  |
| 5     | 0.2680    | 0.3647  | 0.9185    | 0.8874  |
| 6     | 0.2433    | 0.3616  | 0.9258    | 0.8883  |
| 7     | 0.2255    | 0.3574  | 0.9318    | 0.8890  |
| 8     | 0.2138    | 0.3555  | 0.9364    | 0.8914  |
| 9     | 0.2068    | 0.3549  | 0.9388    | 0.8912  |
| 10    | 0.2034    | 0.3549  | 0.9405    | 0.8912  |

---

## Key Observations

- LoRA significantly outperforms head-only finetuning: **89.29% vs 79.89%**
- Best configuration: **Rank=4, Alpha=8** — good balance of parameters vs accuracy
- Higher alpha generally improves performance for same rank
- Rank=8 uses 2.5x more parameters than Rank=2 but gives only marginal improvement
- All LoRA configs converge much faster (epoch 2 already at ~86%) vs baseline

---

## Links
- WandB: https://wandb.ai/sandeshsuman2000-iit-jodhpur/m25csa034_MLDLops
- HuggingFace: https://huggingface.co/sandesh2233/vit-lora-cifar100
