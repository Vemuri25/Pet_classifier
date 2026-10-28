"""Transfer-learning models: an ImageNet-pretrained backbone with a new classification head."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "classifier"),
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(SUPPORTED)}")
    fn, weights, head_attr = SUPPORTED[name]
    model = fn(weights=weights if pretrained else None)

    if head_attr == "fc":  # ResNet
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    else:                  # EfficientNet / MobileNet: replace the last Linear in the classifier
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    model.head_attr = head_attr
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head = getattr(model, model.head_attr)
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head.parameters():
        p.requires_grad = True


def save_checkpoint(path: Path, model: nn.Module, model_name: str, classes, img_size: int, metrics=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": model_name,
                "classes": list(classes), "img_size": img_size, "metrics": metrics or {}}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
