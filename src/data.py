
"""Dataset loading and augmentation for the Oxford-IIIT Pet dataset.

Contains 37 cat and dog breeds.
The dataset is downloaded automatically into the configured data directory.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config import Config


def build_transforms(cfg: Config):
    """Build separate transformations for training and evaluation."""

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.img_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            cfg.imagenet_mean,
            cfg.imagenet_std,
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(cfg.img_size * 1.14)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            cfg.imagenet_mean,
            cfg.imagenet_std,
        ),
    ])

    return train_tf, eval_tf


class TransformSubset(Dataset):
    """Apply a specific transform to selected dataset indices."""

    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]
        image = image.convert("RGB")
        image = self.transform(image)
        return image, label


def get_dataloaders(cfg: Config):
    """Create reproducible training, validation, and test dataloaders."""

    train_tf, eval_tf = build_transforms(cfg)

    full_dataset = datasets.OxfordIIITPet(
        root=cfg.data_dir,
        split="trainval",
        target_types="category",
        download=True,
    )

    test_dataset = datasets.OxfordIIITPet(
        root=cfg.data_dir,
        split="test",
        target_types="category",
        download=True,
        transform=eval_tf,
    )

    classes = full_dataset.classes

    # Reproducible train/validation split.
    generator = torch.Generator().manual_seed(cfg.seed)
    permutation = torch.randperm(
        len(full_dataset),
        generator=generator,
    ).tolist()

    num_validation = int(len(full_dataset) * cfg.val_split)

    val_indices = permutation[:num_validation]
    train_indices = permutation[num_validation:]

    # Optional smaller dataset for quick experiments.
    if cfg.limit > 0:
        train_indices = train_indices[:cfg.limit]
        val_indices = val_indices[
            :max(cfg.limit // 5, 32)
        ]

        test_count = min(
            len(test_dataset),
            max(cfg.limit // 5, 64),
        )

        test_dataset = Subset(
            test_dataset,
            list(range(test_count)),
        )

    train_dataset = TransformSubset(
        full_dataset,
        train_indices,
        train_tf,
    )

    val_dataset = TransformSubset(
        full_dataset,
        val_indices,
        eval_tf,
    )

    # Pin memory is useful for CUDA, but not required for Apple MPS.
    pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": cfg.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        classes,
    )
