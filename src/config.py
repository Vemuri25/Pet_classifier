"""Central configuration. Every value can be overridden from the command line in train.py."""
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


@dataclass
class Config:
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"
    model_name: str = "efficientnet_b0"   # efficientnet_b0 | resnet50 | mobilenet_v3_large
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 8
    freeze_epochs: int = 1                # train only the new head for the first N epochs
    lr: float = 0.0003
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    val_split: float = 0.15
    patience: int = 4                     # early stopping
    num_workers: int = 0
    seed: int = 42
    limit: int = 0                        # >0 = use only N training images (quick CPU test runs)
    imagenet_mean: tuple = field(default=(0.485, 0.456, 0.406))
    imagenet_std: tuple = field(default=(0.229, 0.224, 0.225))
