
"""Train the pet breed classifier using transfer learning."""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import Config, get_device
from src.data import get_dataloaders
from src.model import (
    build_model,
    save_checkpoint,
    set_backbone_trainable,
)


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the best model checkpoint",
    )

    for key in [
        "model_name",
        "epochs",
        "freeze_epochs",
        "batch_size",
        "lr",
        "img_size",
        "num_workers",
        "limit",
        "patience",
        "seed",
    ]:
        parser.add_argument(
            f"--{key}",
            type=type(getattr(cfg, key)),
            default=getattr(cfg, key),
        )

    for key, value in vars(parser.parse_args()).items():
        setattr(cfg, key, value)

    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)

    # During the head-only phase, keep the frozen backbone in eval mode
    # so BatchNorm running statistics are not updated.
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for images, labels in tqdm(
        loader,
        leave=False,
        desc="train" if training else "val",
    ):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def plot_history(history, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    device = get_device()
    print(f"Device: {device} | Model: {cfg.model_name}")

    train_dl, val_dl, _, classes = get_dataloaders(cfg)

    print(
        f"Classes: {len(classes)} | "
        f"Train batches: {len(train_dl)} | "
        f"Validation batches: {len(val_dl)}"
    )

    model = build_model(
        cfg.model_name,
        len(classes),
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.label_smoothing
    )

    # Use a lower learning rate for the pretrained backbone
    # and a higher learning rate for the new classification head.
    head = getattr(model, model.head_attr)
    head_param_ids = {
        id(parameter) for parameter in head.parameters()
    }

    backbone_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": cfg.lr * 0.1,
            },
            {
                "params": head.parameters(),
                "lr": cfg.lr,
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )

    history = {
        key: []
        for key in [
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
        ]
    }

    best_acc = 0.0
    bad_epochs = 0

    from pathlib import Path

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output:
        checkpoint_path = Path(cfg.output)
    else:
        checkpoint_path = cfg.artifacts_dir / "model.pt"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        # Initially train only the classification head.
        # Then unfreeze the backbone for gentle fine-tuning.
        frozen = epoch <= cfg.freeze_epochs

        set_backbone_trainable(
            model,
            trainable=not frozen,
        )

        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model,
            train_dl,
            criterion,
            device,
            optimizer,
            scaler,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_dl,
            criterion,
            device,
        )

        scheduler.step()

        values = [
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ]

        for key, value in zip(history, values):
            history[key].append(value)

        elapsed = time.time() - start_time
        phase = "[head only] " if frozen else ""

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} {phase}"
            f"Train loss: {train_loss:.3f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.3f} | "
            f"Val acc: {val_acc:.3f} | "
            f"{elapsed:.0f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0

            save_checkpoint(
                checkpoint_path,
                model,
                cfg.model_name,
                classes,
                cfg.img_size,
                {
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
            )

            print(
                f"  Saved best model "
                f"(validation accuracy: {val_acc:.3f}) "
                f"-> {checkpoint_path}"
            )
        else:
            bad_epochs += 1

            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    history_path = cfg.artifacts_dir / "history.json"
    history_path.write_text(
        json.dumps(history, indent=2)
    )

    plot_history(
        history,
        cfg.artifacts_dir / "training_curves.png",
    )

    print(
        f"Training complete. Best validation accuracy: "
        f"{best_acc:.3f}"
    )
    print("Training curves: artifacts/training_curves.png")


if __name__ == "__main__":
    main()
