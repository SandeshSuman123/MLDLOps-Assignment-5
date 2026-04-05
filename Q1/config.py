import os

#  Paths 
DATA_DIR   = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  WandB / HuggingFace
WANDB_PROJECT  = "DLOps-Ass5-Q1-ViT-LoRA"
WANDB_ENTITY   = os.environ.get("WANDB_ENTITY", None)   # set via env or leave None
HF_REPO_ID     = os.environ.get("HF_REPO_ID", "")       # e.g. "yourname/vit-lora-cifar100"

#  Dataset
NUM_CLASSES    = 100
IMAGE_SIZE     = 224       # ViT-S expects 224×224
BATCH_SIZE     = 512        
NUM_WORKERS    = 12

# Training
EPOCHS         = 10
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
SEED           = 42

# LoRA search space
LORA_RANKS     = [2, 4, 8]
LORA_ALPHAS    = [2, 4, 8]
LORA_DROPOUT   = 0.1
LORA_TARGETS   = ["qkv"]   # ViT attention projections in timm

# Optuna 
OPTUNA_TRIALS  = 9
OPTUNA_EPOCHS  = 2          # shorter runs for HPO