"""
optuna_search.py
Uses Optuna to find the best LoRA rank + alpha 
then retrains best config for full EPOCHS and saves to HuggingFace.

Usage:
    python optuna_search.py
"""

import os, json, torch
import optuna

import wandb
import config
from dataset  import get_loaders
from model    import build_lora, count_trainable
from trainer  import train
from evaluate import evaluate


def objective(trial, train_loader, val_loader, device):
    rank    = trial.suggest_categorical("rank",    config.LORA_RANKS)
    alpha   = trial.suggest_categorical("alpha",   config.LORA_ALPHAS)
    lr      = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])

    model = build_lora(rank=rank, alpha=alpha, dropout=dropout)
    run_name = f"optuna_trial{trial.number}_r{rank}_a{alpha}"

    _, best_val = train(
        model, train_loader, val_loader,
        run_name=run_name,
        epochs=config.OPTUNA_EPOCHS,
        lr=lr,
        device=device,
        use_wandb=True,
        wandb_cfg={
            "optuna_trial": trial.number,
            "rank": rank, "alpha": alpha,
            "dropout": dropout, "lr": lr,
        },
        log_grad=False,
    )
    return best_val


def main():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_loaders()

    study = optuna.create_study(
        direction="maximize",
        study_name="vit_lora_cifar100",
        storage=f"sqlite:///{config.OUTPUT_DIR}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )

    study.optimize(
        lambda t: objective(t, train_loader, val_loader, device),
        n_trials=config.OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nBest params: {best}")
    print(f"Best val acc: {study.best_value:.4f}")

    with open(f"{config.OUTPUT_DIR}/optuna_best_params.json", "w") as f:
        json.dump({"best_params": best, "best_val_acc": study.best_value}, f, indent=2)

    # ── Retrain best config for full epochs ───────────────────────────────────
    print("\nRetraining best config for full epochs...")
    best_model = build_lora(
        rank=best["rank"], alpha=best["alpha"], dropout=best.get("dropout", 0.1)
    )
    history, _ = train(
        best_model, train_loader, val_loader,
        run_name="optuna_best_full",
        epochs=config.EPOCHS,
        lr=best.get("lr", config.LR),
        device=device,
        use_wandb=True,
        wandb_cfg={"mode": "best_retrain", **best},
        log_grad=True,
    )

    # Save locally
    ckpt_path = f"{config.OUTPUT_DIR}/best_model.pth"
    torch.save(best_model.state_dict(), ckpt_path)

    # Test
    wandb.init(project=config.WANDB_PROJECT, name="best_model_test", reinit="finish_previous")
    test_acc, _ = evaluate(best_model, test_loader, device, "best_model")
    wandb.finish()

    # Push to HuggingFace Hub 
    hf_repo = config.HF_REPO_ID
    if hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo="best_model.pth",
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"Best LoRA model — test_acc={test_acc:.4f}",
            )
            # Upload config card
            card_path = f"{config.OUTPUT_DIR}/README_hf.md"
            with open(card_path, "w") as f:
                f.write(f"""---
license: mit
tags:
- image-classification
- lora
- vit
- cifar100
---
# ViT-S + LoRA — CIFAR-100

**Best LoRA config found by Optuna:**
- Rank: {best['rank']}
- Alpha: {best['alpha']}
- Dropout: {best.get('dropout', 0.1)}
- LR: {best.get('lr', config.LR):.2e}

**Test Accuracy: {test_acc:.4f}**
""")
            api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=hf_repo,
                repo_type="model",
            )
            print(f"Pushed to HuggingFace: {hf_repo}")
        except Exception as e:
            print(f"HuggingFace upload failed: {e}")
    else:
        print("Set HF_REPO_ID env var to push to HuggingFace.")

    print(f"\nDone! Best test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()