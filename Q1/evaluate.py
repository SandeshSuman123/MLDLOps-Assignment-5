"""
evaluate.py
- Overall test accuracy
- Per-class accuracy histogram (saved + logged to WandB)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import config


CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle',
    'bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle',
    'caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch',
    'crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest',
    'fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower',
    'leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear',
    'pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum',
    'rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew',
    'skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower',
    'sweet_pepper','table','tank','telephone','television','tiger','tractor','train',
    'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]


@torch.no_grad()
def evaluate(model, loader, device, run_name: str, use_wandb: bool = True):
    model.eval().to(device)
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="Testing"):
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    overall_acc = (preds == labels).mean()
    print(f"[{run_name}] Test Accuracy: {overall_acc:.4f}")

    # Per-class accuracy
    class_acc = np.zeros(config.NUM_CLASSES)
    for c in range(config.NUM_CLASSES):
        mask = labels == c
        class_acc[c] = (preds[mask] == labels[mask]).mean() if mask.sum() > 0 else 0.0

    # Plot histogram
    fig, ax = plt.subplots(figsize=(24, 5))
    bars = ax.bar(range(config.NUM_CLASSES), class_acc * 100, color="steelblue")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Class-wise Test Accuracy — {run_name}")
    ax.set_ylim(0, 100)
    plt.tight_layout()

    hist_path = f"{config.OUTPUT_DIR}/{run_name}_classwise.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    if use_wandb:
        wandb.log({
            f"test/overall_accuracy": overall_acc,
            f"test/classwise_histogram": wandb.Image(hist_path),
        })

    return overall_acc, class_acc