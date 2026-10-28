"""Inference helper used by the CLI and the API.

    python -m src.predict path/to/image.jpg
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.model import load_checkpoint


class Predictor:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.model, ckpt = load_checkpoint(checkpoint_path, device)
        self.classes = ckpt["classes"]
        cfg = Config()
        size = ckpt["img_size"]
        self.tf = transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(cfg.imagenet_mean, cfg.imagenet_std),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5):
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        top = torch.topk(probs, k=min(top_k, len(self.classes)))
        return [{"label": self.classes[i], "probability": round(float(p), 4)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--checkpoint", default=str(Config().artifacts_dir / "model.pt"))
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    pred = Predictor(args.checkpoint)
    for r in pred.predict(Image.open(Path(args.image)), args.top_k):
        print(f"{r['label']:<30} {r['probability']:.2%}")


if __name__ == "__main__":
    main()
