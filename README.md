# Pet Breed Classifier

A PyTorch computer-vision project that classifies images into **37 cat and dog breeds** using transfer learning with **EfficientNet-B0**. The model is served through a **FastAPI** application with a browser-based upload interface and top-5 predictions.

- **Live demo:** https://pet-classifier-yigd.onrender.com
- **GitHub repository:** https://github.com/Vemuri25/Pet_classifier
- **Dataset:** Oxford-IIIT Pet Dataset
- **Preserved production checkpoint:** `artifacts/model.pt`
- **Reported test accuracy:** **90.79%** (3,331 / 3,669 test images)

> The production checkpoint is the preserved baseline. Experimental checkpoints and reports should be treated as comparisons, not replacements for the production model.

## Highlights

- ImageNet-pretrained EfficientNet-B0 transfer learning
- 37-class pet-breed classification
- PyTorch training and evaluation pipeline
- FastAPI inference endpoints and browser-based interface
- Top-k predictions with class probabilities
- Docker support
- pytest tests and GitHub Actions CI
- Experiment reports and error analysis

## Architecture

```text
User uploads an image
        |
        v
FastAPI web interface / API
        |
        v
Image preprocessing (224 x 224)
        |
        v
EfficientNet-B0 backbone
(ImageNet pretrained)
        |
        v
37-class classification head
        |
        v
Class probabilities
        |
        v
Top-5 breed predictions
```

## Results

The preserved production EfficientNet-B0 model achieved **90.79% test accuracy** on the official Oxford-IIIT Pet test split (3,331 correct predictions out of 3,669 images).

| Experiment | Test accuracy | Outcome |
|---|---:|---|
| Production EfficientNet-B0 | **90.79%** | Preserved production checkpoint |
| Augmented EfficientNet-B0 candidate | 89.72% | Lower overall test accuracy |
| EfficientNet-B0 candidate | 88.12% | Lower overall test accuracy |
| EfficientNet-B0 with BatchNorm adjustment | 88.58% | Lower overall test accuracy |
| ResNet-50 candidate | 90.30% | Lower than production model |

The production model's error analysis identified challenging classes including American Pit Bull Terrier, Staffordshire Bull Terrier, Ragdoll, and Russian Blue. Future improvements should be evaluated per class as well as by overall accuracy.

Metrics and visual reports are stored under `artifacts/`; experiment reports are under `experiments/` where available. Training and evaluation commands may overwrite checkpoints, metrics, and plots, so back up any artifacts you want to preserve first.

## Repository structure

```text
pet-classifier/
├── src/
│   ├── config.py          # Configuration and device selection
│   ├── data.py            # Dataset loading and preprocessing
│   ├── model.py           # Model definitions and checkpoint utilities
│   ├── train.py           # Training and fine-tuning
│   ├── evaluate.py        # Test evaluation and confusion matrix
│   ├── predict.py         # Prediction utility
│   └── error_analysis.py  # Error-analysis utilities
├── serving/
│   ├── app.py             # FastAPI application
│   └── index.html         # Browser-based interface
├── tests/                 # Model and API tests
├── artifacts/
│   ├── model.pt           # Preserved production checkpoint
│   ├── metrics.json       # Evaluation metrics
│   ├── history.json       # Training history
│   ├── confusion_matrix.png
│   └── training_curves.png
├── Dockerfile
├── requirements.txt
├── requirements-serve.txt
└── README.md
```

## Requirements

- Python 3.10–3.12 recommended
- Git
- CPU or supported GPU
- Docker Desktop (optional)

## Run locally on macOS or Linux

```bash
git clone https://github.com/Vemuri25/Pet_classifier.git
cd Pet_classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the tests:

```bash
pytest -q
```

The project previously passed with **8 tests passing**. Results can vary if dependencies or code have changed.

## Start the API

From the project root:

```bash
uvicorn serving.app:app --reload --port 8000
```

- Web interface: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

Example API request:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \\
  -F "file=@path/to/your_pet_image.jpg"
```

Replace the example image path with the path to an image on your computer.

## Train and evaluate

**Important:** Training and evaluation may overwrite files in `artifacts/`, including checkpoints, metrics, and plots. Back up the existing production checkpoint and reports before running these commands. Retraining is not required just to use the deployed app.

```bash
python -m src.train
python -m src.evaluate
python -m src.predict path/to/your_pet_image.jpg
```

For a smaller sanity run, if supported by the current training script:

```bash
python -m src.train --limit 400 --epochs 2 --num_workers 0
```

Check the current command-line options with:

```bash
python -m src.train --help
python -m src.evaluate --help
python -m src.predict --help
```

## Docker

```bash
docker build -t pet-classifier .
docker run --rm -p 8000:8000 pet-classifier
```

Open http://localhost:8000 in your browser.

## Deployment

The application is deployed on Render:

**Live application:** https://pet-classifier-yigd.onrender.com

On free hosting tiers, the service may sleep while idle, so the first request after inactivity can take longer.

## Technologies

Python · PyTorch · Torchvision · EfficientNet-B0 · FastAPI · Uvicorn · Docker · pytest · GitHub Actions

## Potential next steps

- Add Grad-CAM visualizations to explain image predictions
- Compare per-class precision, recall, and F1 across model candidates
- Export to ONNX and benchmark inference latency
- Track experiments with MLflow or Weights & Biases
- Explore model quantization and inference optimizations
- Monitor prediction latency and common misclassifications

## Troubleshooting

| Issue | Suggested check |
|---|---|
| `ModuleNotFoundError` for `src` | Run commands from the repository root and use `python -m src.train`. |
| API cannot load the model | Confirm the checkpoint exists at the configured path and inspect server logs. |
| `CUDA out of memory` | Reduce batch size or use a smaller model/input configuration. |
| Dataset download fails | Check connectivity and retry the download. |
| First request on Render is slow | The hosted service may be waking from an idle state. |

## License and dataset

Review the dataset's terms and the repository's license before redistributing the dataset or model artifacts. This project uses the Oxford-IIIT Pet Dataset for pet-breed classification.
