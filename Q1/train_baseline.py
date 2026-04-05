"""
train_baseline.py
Finetune ViT-S classification head only (no LoRA) on CIFAR-100.

Usage:
    python train_baseline.py
"""

import os, torch, json
import wandb
import config
from dataset import get_loaders
from model   import build_baseline, count_trainable, print_trainable
from trainer import train
from evaluate import evaluate


def main():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_loaders()

    model = build_baseline()
    print_trainable(model, "Baseline")

    run_name = "baseline_head_only"
    history, best_val = train(
        model, train_loader, val_loader,
        run_name=run_name,
        epochs=config.EPOCHS,
        device=device,
        use_wandb=True,
        wandb_cfg={"mode": "baseline", "trainable_params": count_trainable(model)},
        log_grad=False,
    )

    # Save weights
    ckpt_path = f"{config.OUTPUT_DIR}/{run_name}_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Test
    wandb.init(project=config.WANDB_PROJECT, name=f"{run_name}_test", reinit=True)
    test_acc, _ = evaluate(model, test_loader, device, run_name)
    wandb.finish()

    # Save results JSON
    results = {
        "run": run_name,
        "lora": False,
        "rank": None, "alpha": None, "dropout": None,
        "trainable_params": count_trainable(model),
        "test_accuracy": float(test_acc),
        "history": history,
    }
    with open(f"{config.OUTPUT_DIR}/{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done — baseline.")


if __name__ == "__main__":
    main()