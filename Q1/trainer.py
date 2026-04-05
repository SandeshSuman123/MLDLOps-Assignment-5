"""
trainer.py
Core training / evaluation loop with:
  - Mixed-precision (AMP) for speed & memory
  - WandB logging (loss, accuracy, LoRA gradient norms)
  - Returns per-epoch metrics dict
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm
import wandb
import config


# Helpers 
def _lora_grad_norms(model: nn.Module) -> dict:
    """Return gradient norms for LoRA A/B matrices (for WandB)."""
    norms = {}
    for name, p in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and p.grad is not None:
            norms[f"grad_norm/{name}"] = p.grad.norm().item()
    return norms


def run_epoch(model, loader, optimizer, criterion, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False,
                                 desc="train" if train else "val  "):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.autocast("cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


#  Main training function 
def train(
    model,
    train_loader,
    val_loader,
    *,
    run_name: str,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    device: torch.device,
    use_wandb: bool = True,
    wandb_cfg: dict = None,
    log_grad: bool = True,
):
    """
    Train model and return list of per-epoch metric dicts.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    if use_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=run_name,
            config=wandb_cfg or {},
            reinit="finish_previous",
            mode="offline"  
        )

    history = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion,
                                    scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion,
                                      scaler, device, train=False)
        scheduler.step()

        row = dict(epoch=epoch,
                   train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                   val_loss=round(val_loss, 4),  val_acc=round(val_acc, 4))
        history.append(row)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f} | "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f}")

        if use_wandb:
            log = {
                "epoch": epoch,
                "train/loss": tr_loss,  "train/accuracy": tr_acc,
                "val/loss": val_loss,   "val/accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            if log_grad:
                log.update(_lora_grad_norms(model))
            wandb.log(log)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    if use_wandb:
        wandb.finish()

    return history, best_val_acc