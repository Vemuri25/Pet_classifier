# 🐾 Pet Breed Classifier — End-to-End Deep Learning Project

Transfer-learning image classifier (EfficientNet-B0, PyTorch) that recognises **37 cat and dog breeds**
from the Oxford-IIIT Pet dataset, served through a **FastAPI** REST API with a drag-and-drop web UI,
containerised with **Docker**, tested with **pytest**, and checked by **GitHub Actions CI**.

> Resume line: *Built and deployed a PyTorch transfer-learning classifier for 37 pet breeds
> (≈90% test accuracy) with a FastAPI inference service, Docker image and CI pipeline; live demo on Hugging Face Spaces.*
> (Replace the accuracy with the number your own `artifacts/metrics.json` reports.)

---

## Project structure

```
pet-classifier/
├── src/
│   ├── config.py        # all hyper-parameters + device selection (CUDA / Apple MPS / CPU)
│   ├── data.py          # dataset download, train/val split, augmentation
│   ├── model.py         # pretrained backbone + new head, freeze/unfreeze, save/load
│   ├── train.py         # training loop: AMP, AdamW, cosine LR, early stopping, curves
│   ├── evaluate.py      # test-set report, macro-F1, confusion matrix
│   └── predict.py       # Predictor class + CLI inference
├── serving/
│   ├── app.py           # FastAPI: /, /health, /predict, /docs
│   └── index.html       # web UI
├── tests/               # pytest: model shapes, freezing logic, API endpoints
├── artifacts/           # created by training: model.pt, curves, metrics, confusion matrix
├── .vscode/             # run/debug configs for VS Code
├── .github/workflows/   # CI
├── Dockerfile
├── requirements.txt         # training + dev
└── requirements-serve.txt   # slim deps for the Docker image
```

---

## Step 0 — Prerequisites

| Tool | Get it |
|---|---|
| Python 3.10–3.12 | python.org (Windows: tick **“Add python.exe to PATH”**) |
| VS Code + **Python** extension (Microsoft) | code.visualstudio.com |
| Git | git-scm.com |
| Docker Desktop (for Step 7) | docker.com |
| NVIDIA GPU (optional) | training on CPU works with `--limit`; a GPU or free Google Colab/Kaggle GPU is better for the full run |

## Step 1 — Open the project in VS Code

Unzip the folder, then in VS Code: **File → Open Folder… → pet-classifier**.
Open the terminal with **Ctrl + `** (backtick).

## Step 2 — Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If you get "running scripts is disabled": Set-ExecutionPolicy -Scope CurrentUser RemoteSigned  then retry
```
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Your prompt now starts with `(.venv)`. In VS Code press **Ctrl+Shift+P → “Python: Select Interpreter”** and pick `.venv`.

## Step 3 — Install dependencies

**Install PyTorch first, matching your hardware:**
```bash
# CPU only (any laptop)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU — get the exact command for your CUDA version from https://pytorch.org/get-started/locally/
# Apple Silicon Mac — plain install, MPS is used automatically
pip install torch torchvision
```
Then everything else:
```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## Step 4 — Run the tests (checks the setup before you spend time training)
```bash
pytest -q
```
Expected: `8 passed`.

## Step 5 — Train

**Quick sanity run (≈5–10 min on CPU)** — downloads the dataset (~800 MB, first time only):
```bash
python -m src.train --limit 400 --epochs 2 --num_workers 0
```
**Full run (GPU: ~10–15 min; CPU: a few hours):**
```bash
python -m src.train
```
Useful options: `--model_name resnet50`, `--epochs 15`, `--batch_size 16` (if you run out of GPU memory),
`--num_workers 0` (if Windows complains about worker processes).

What happens:
1. Epochs 1–2 train only the new classification head (backbone frozen).
2. Then the whole network is fine-tuned with a lower learning rate.
3. The best model by validation accuracy is saved to `artifacts/model.pt`; training stops early if it stops improving.
4. `artifacts/training_curves.png` shows loss/accuracy curves.

> No GPU? Upload the folder to Google Colab or Kaggle, choose a GPU runtime, run Steps 3 and 5 there,
> then download `artifacts/model.pt` back into your local `artifacts/` folder.

## Step 6 — Evaluate and predict
```bash
python -m src.evaluate                    # precision/recall/F1 per breed, confusion matrix
python -m src.predict path/to/your_dog.jpg
```
Outputs `artifacts/metrics.json` and `artifacts/confusion_matrix.png` — put both images in your GitHub README.

## Step 7 — Serve the model locally
```bash
uvicorn serving.app:app --reload --port 8000
```
- Web UI → http://localhost:8000
- Interactive API docs → http://localhost:8000/docs
- Health check → http://localhost:8000/health

API from the terminal:
```bash
curl -X POST "http://localhost:8000/predict?top_k=3" -F "file=@my_cat.jpg"
```
(VS Code alternative: **Run and Debug** panel → pick *Train*, *Evaluate* or *Run API* → ▶.)

## Step 8 — Docker
```bash
docker build -t pet-classifier .
docker run -p 8000:8000 pet-classifier
```
Open http://localhost:8000 — the same app, now running in a container.

## Step 9 — Push to GitHub (CI runs automatically)
```bash
git init
git lfs install                 # model.pt is ~16 MB; LFS keeps the repo light
git lfs track "*.pt"
git add .
git commit -m "Pet breed classifier: training, API, Docker, CI"
git branch -M main
git remote add origin https://github.com/<your-username>/pet-classifier.git
git push -u origin main
```
The **Actions** tab will show the test workflow running on every push.

## Step 10 — Deploy for free (a live link on your resume)

### Option A: Hugging Face Spaces (Docker)
1. Create an account at huggingface.co → **New Space** → SDK: **Docker** → Blank.
2. Spaces route traffic to port 7860, so add these lines to the **top** of the README.md you push to the Space:
   ```yaml
   ---
   title: Pet Breed Classifier
   emoji: 🐾
   sdk: docker
   app_port: 7860
   ---
   ```
   and in the Dockerfile change `PORT=8000` to `PORT=7860`.
3. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/pet-classifier
   git push space main
   ```
   (Use a Hugging Face access token with write permission as the password.)
4. The Space builds the Docker image and gives you a public URL.

### Option B: Render (or Railway / Google Cloud Run)
Create a **Web Service** from your GitHub repo, choose **Docker** as the runtime; the app reads the `PORT`
environment variable the platform provides. Free tiers sleep when idle, so the first request may be slow.

---

## Ideas to extend it (and talk about in interviews)
- **Grad-CAM** heatmaps to show *where* the model looks.
- **Export to ONNX** and compare CPU latency against PyTorch.
- **Experiment tracking** with MLflow or Weights & Biases; compare EfficientNet vs ResNet50 vs MobileNet.
- **Quantisation** (`torch.ao.quantization`) for a smaller, faster model.
- **Monitoring**: log predictions and latency; add Prometheus metrics.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root, using `python -m src.train` (not `python src/train.py`). |
| Windows DataLoader hangs / `BrokenPipeError` | Add `--num_workers 0`. |
| `CUDA out of memory` | `--batch_size 16` or `8`. |
| Dataset download fails | Re-run; partial downloads resume. Or delete `data/` and retry. |
| API says “Model not loaded” | Train first so `artifacts/model.pt` exists, or set `MODEL_PATH`. |
