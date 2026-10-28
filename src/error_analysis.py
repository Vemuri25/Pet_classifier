import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config import Config, get_device
from src.data import get_dataloaders
from src.model import load_checkpoint


def main():
    cfg = Config()
    device = get_device()

    model, checkpoint = load_checkpoint(
        str(cfg.artifacts_dir / "model.pt"), device
    )
    cfg.img_size = checkpoint["img_size"]

    _, _, test_loader, classes = get_dataloaders(cfg)

    predictions, labels = [], []

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Analyzing test set"):
            outputs = model(images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(targets.numpy())

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    cm = confusion_matrix(
        labels, predictions, labels=np.arange(len(classes))
    )

    mistakes = []
    for true_idx, row in enumerate(cm):
        for pred_idx, count in enumerate(row):
            if true_idx != pred_idx and count > 0:
                mistakes.append(
                    (int(count), classes[true_idx], classes[pred_idx])
                )

    mistakes.sort(reverse=True)

    print("\nTOP 20 BREED CONFUSIONS")
    print("-" * 65)
    for count, actual, predicted in mistakes[:20]:
        print(f"{count:4d}  Actual: {actual:<30} Predicted: {predicted}")

    print("\nLOWEST RECALL BREEDS")
    print("-" * 65)
    for idx, breed in enumerate(classes):
        total = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        recall = correct / total if total else 0.0
        print(
            f"{recall:.3f} recall | {correct:3d}/{total:3d} correct | {breed}"
        )

    total_errors = int((predictions != labels).sum())
    accuracy = float((predictions == labels).mean())
    print(f"\nTest images: {len(labels)}")
    print(f"Incorrect predictions: {total_errors}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
