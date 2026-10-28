"""Evaluate the best checkpoint on the held-out test set.

    python -m src.evaluate
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import Config, get_device
from src.data import get_dataloaders
from src.model import load_checkpoint


def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(cfg.artifacts_dir / "model.pt"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    cfg.limit = args.limit

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg.img_size = ckpt["img_size"]
    _, _, test_dl, classes = get_dataloaders(cfg)

    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_dl, desc="test"):
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    acc = float((preds == labels).mean())
    report = classification_report(labels, preds, labels=range(len(classes)),
                                   target_names=classes, output_dict=True, zero_division=0)
    print(classification_report(labels, preds, labels=range(len(classes)),
                                target_names=classes, zero_division=0))
    print(f"Test accuracy: {acc:.4f}")

    cfg.artifacts_dir.mkdir(exist_ok=True)
    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(
        {"test_accuracy": acc, "macro_f1": report["macro avg"]["f1-score"], "per_class": report}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(labels, preds, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix (acc {acc:.3f})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(cfg.artifacts_dir / "confusion_matrix.png", dpi=120)
    print("Saved artifacts/metrics.json and artifacts/confusion_matrix.png")


if __name__ == "__main__":
    main()
