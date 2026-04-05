"""
train_lora.py
Runs all 9 LoRA combinations (rank ∈ {2,4,8} × alpha ∈ {2,4,8})
sequentially, logging each to WandB and saving weights.

Usage:
    python train_lora.py
    python train_lora.py --rank 4 --alpha 4   # single experiment
"""

import argparse, json, os, torch
import wandb
import config
from dataset  import get_loaders
from model    import build_lora, count_trainable, print_trainable
from trainer  import train
from evaluate import evaluate


def run_experiment(rank, alpha, train_loader, val_loader, test_loader, device, exp_no):
    run_name = f"lora_r{rank}_a{alpha}_d{config.LORA_DROPOUT}"
    print(f"\n{'='*60}")
    print(f"Experiment {exp_no}: rank={rank}, alpha={alpha}, dropout={config.LORA_DROPOUT}")
    print(f"{'='*60}")

    model = build_lora(rank=rank, alpha=alpha)
    print_trainable(model, run_name)

    history, best_val = train(
        model, train_loader, val_loader,
        run_name=run_name,
        epochs=config.EPOCHS,
        device=device,
        use_wandb=True,
        wandb_cfg={
            "rank": rank, "alpha": alpha,
            "dropout": config.LORA_DROPOUT,
            "lora_targets": config.LORA_TARGETS,
            "trainable_params": count_trainable(model),
        },
        log_grad=True,
    )

    # Save weights
    ckpt_path = f"{config.OUTPUT_DIR}/{run_name}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Test evaluation
    wandb.init(project=config.WANDB_PROJECT, name=f"{run_name}_test", reinit=True)
    test_acc, _ = evaluate(model, test_loader, device, run_name)
    wandb.finish()

    results = {
        "exp_no": exp_no, "run": run_name, "lora": True,
        "rank": rank, "alpha": alpha, "dropout": config.LORA_DROPOUT,
        "trainable_params": count_trainable(model),
        "best_val_acc": float(best_val),
        "test_accuracy": float(test_acc),
        "history": history,
    }
    with open(f"{config.OUTPUT_DIR}/{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",  type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_loaders()

    # Build experiment grid
    if args.rank and args.alpha:
        grid = [(args.rank, args.alpha)]
    else:
        grid = [(r, a) for r in config.LORA_RANKS for a in config.LORA_ALPHAS]

    all_results = []
    for exp_no, (rank, alpha) in enumerate(grid, 1):
        res = run_experiment(rank, alpha, train_loader, val_loader,
                             test_loader, device, exp_no)
        all_results.append(res)

    # Summary table
    print("\n\n" + "="*70)
    print(f"{'Exp':>4} {'Rank':>5} {'Alpha':>6} {'Dropout':>8} "
          f"{'Test Acc':>10} {'Trainable':>12}")
    print("-"*70)
    for r in all_results:
        print(f"{r['exp_no']:>4} {r['rank']:>5} {r['alpha']:>6} "
              f"{r['dropout']:>8} {r['test_accuracy']:>10.4f} "
              f"{r['trainable_params']:>12,}")

    with open(f"{config.OUTPUT_DIR}/all_lora_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll results saved.")


if __name__ == "__main__":
    main()