"""FastAPI service + a small web UI.

    uvicorn serving.app:app --reload --port 8000
    open http://localhost:8000   (UI)   |   http://localhost:8000/docs   (Swagger)
"""
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "model.pt"))
    if path.exists():
        STATE["predictor"] = Predictor(path)
        print(f"Loaded model from {path}")
    else:
        print(f"WARNING: no model at {path}. Train first: python -m src.train")
    yield
    STATE.clear()


app = FastAPI(title="Pet Breed Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in STATE}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if "predictor" not in STATE:
        raise HTTPException(503, "Model not loaded")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, "File is not a valid image")
    t0 = time.perf_counter()
    preds = STATE["predictor"].predict(img, top_k)
    return {"predictions": preds, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
