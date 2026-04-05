"""
model.py
Builds:
  - Baseline ViT-S with only classification head trainable
  - LoRA-wrapped ViT-S (Q, K, V) with trainable head
"""

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import config


#  Map timm ViT attention layer names 
# timm's vit_small_patch16_224 names the QKV proj as:
#   blocks.X.attn.qkv   (merged)  OR
#   blocks.X.attn.q_proj / k_proj / v_proj (split)
# We target the split projections via PEFT module names.
TIMM_QKV_MODULES = ["qkv"]


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the timm classification head."""
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def build_baseline() -> nn.Module:
    """ViT-S pretrained, only head trainable."""
    model = timm.create_model(
        "vit_small_patch16_224", pretrained=True, num_classes=config.NUM_CLASSES
    )
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def build_lora(rank: int, alpha: int, dropout: float = config.LORA_DROPOUT) -> nn.Module:
    """ViT-S pretrained + LoRA on Q,K,V + trainable head."""
    model = timm.create_model(
        "vit_small_patch16_224", pretrained=True, num_classes=config.NUM_CLASSES
    )

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=TIMM_QKV_MODULES,
        lora_dropout=dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Ensure classification head stays trainable
    for name, p in model.named_parameters():
        if "head" in name:
            p.requires_grad = True

    return model


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable(model: nn.Module, tag: str = ""):
    total   = sum(p.numel() for p in model.parameters())
    train   = count_trainable(model)
    print(f"[{tag}] Trainable: {train:,} / {total:,} ({100*train/total:.2f}%)")