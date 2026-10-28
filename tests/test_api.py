import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.model import build_model, save_checkpoint


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # A random-weight checkpoint is enough to test the API plumbing.
    classes = [f"breed_{i}" for i in range(37)]
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, build_model("mobilenet_v3_large", 37, pretrained=False),
                    "mobilenet_v3_large", classes, 224)
    monkeypatch.setenv("MODEL_PATH", str(ckpt))
    from serving.app import app
    with TestClient(app) as c:
        yield c


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (300, 260), (120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    r = client.post("/predict?top_k=3", files={"file": ("cat.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3 and preds[0]["probability"] >= preds[-1]["probability"]


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_ui_served(client):
    assert "Pet Breed Classifier" in client.get("/").text
