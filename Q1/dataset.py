import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config


def get_transforms(train: bool):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_loaders():
    train_ds = datasets.CIFAR100(config.DATA_DIR, train=True,
                                  download=True, transform=get_transforms(True))
    test_ds  = datasets.CIFAR100(config.DATA_DIR, train=False,
                                  download=True, transform=get_transforms(False))

    # 80/20 train-val split
    val_size   = int(0.2 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    # val split uses test-time transforms
    val_ds.dataset.transform = get_transforms(False)

    kwargs = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                  pin_memory=True,persistent_workers=True, prefetch_factor=3)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )