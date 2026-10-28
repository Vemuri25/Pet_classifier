import pytest
import torch

from src.model import build_model, set_backbone_trainable


@pytest.mark.parametrize("name", ["efficientnet_b0", "resnet50", "mobilenet_v3_large"])
def test_forward_shape(name):
    model = build_model(name, num_classes=37, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)


def test_freeze_keeps_head_trainable():
    model = build_model("efficientnet_b0", 37, pretrained=False)
    set_backbone_trainable(model, False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("classifier") for n in trainable)
