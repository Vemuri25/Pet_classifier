https://petclassifier-zgg5u5pnffprkygz6ipu8f.streamlit.app/# Stanford Dogs Breed Classifier

A deep learning application that classifies images into **120 dog breeds** using transfer learning with **EfficientNet-B0** and PyTorch.

🚀 **[Launch Live Application](https://petclassifier-zgg5u5pnnfprkygz6ipu8f.streamlit.app/)**

**[GitHub Repository](https://github.com/Vemuri25/Pet_classifier)**

---

## Project Overview

This project implements a 120-class dog breed image classification system using the Stanford Dogs Dataset.

The application allows users to upload a dog photograph and receive:
- Predicted dog breed
- Model confidence score
- Top three breed predictions

The trained model is deployed through Streamlit Community Cloud, providing an interactive web interface for image classification.

## Model Performance

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | **82.03%** |
| Macro F1-Score | **81.56%** |
| Number of Classes | **120** |
| Model Architecture | EfficientNet-B0 |
| Input Image Size | 224 × 224 |

These metrics represent evaluation results on the project's test split. Individual prediction confidence scores are not equivalent to overall model accuracy.

## Application Preview

Users can upload an image and receive the model's three most likely breed predictions with corresponding confidence scores.

## System Architecture

```text
        User
         |
         v
   Upload Dog Image
         |
         v
   Image Preprocessing
      (224 x 224)
         |
         v
   EfficientNet-B0
   Feature Extractor
         |
         v
   120-Class Classification
         |
         v
   Softmax Probabilities
         |
         v
   Top-3 Breed Predictions
         |
         v
   Streamlit Interface
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision |
| Model Architecture | EfficientNet-B0 |
| Transfer Learning | ImageNet Pretrained Weights |
| Web Application | Streamlit |
| Image Processing | Pillow |
| Deployment | Streamlit Community Cloud |
| Version Control | Git and GitHub |

## Dataset

**Stanford Dogs Dataset**

The dataset contains images representing 120 different dog breeds.

The project uses a cleaned and organized dataset with separate training, validation, and testing partitions.

| Dataset Split | Images |
|---|---:|
| Training | 14,265 |
| Validation | 3,006 |
| Testing | 3,183 |
| **Total** | **20,454** |

The dataset was cleaned to remove invalid images and cross-split duplicate leakage.

## Model Development

The project uses transfer learning with an ImageNet-pretrained EfficientNet-B0 architecture.

The classification head was adapted to predict 120 dog breeds.

The V2 training configuration includes:
- EfficientNet-B0 backbone
- Dropout regularization (0.3)
- Cross-entropy loss with label smoothing
- AdamW optimizer
- Learning rate of 0.0001
- Weight decay of 0.0005
- Validation-based checkpoint selection

The selected V2 checkpoint is preserved at:

`artifacts/model_expanded_v2_candidate.pt`

## Project Structure

```text
pet-classifier/
│
├── app.py
│
├── artifacts/
│   ├── model.pt
│   └── model_expanded_v2_candidate.pt
│
├── data/
│   └── expanded_v4/
│
├── src/
│   ├── config.py
│   ├── model.py
│   ├── expanded_data.py
│   ├── train_expanded_v2.py
│   ├── evaluate_expanded_v2.py
│   └── predict.py
│
├── tests/
│
├── requirements.txt
├── requirements-serve.txt
├── Dockerfile
└── README.md
```

The deployed Streamlit application uses `artifacts/model_expanded_v2_candidate.pt`. The older `artifacts/model.pt` checkpoint belongs to the original project and is not the deployed 120-class model.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vemuri25/Pet_classifier.git

cd Pet_classifier
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Open:

`http://localhost:8501`

Upload a dog image to test the classifier.

## Deployment

The application is hosted on Streamlit Community Cloud.

**Live Application:**

https://petclassifier-zgg5u5pnnfprkygz6ipu8f.streamlit.app/

Deployment configuration:

- Repository: `Vemuri25/Pet_classifier`
- Branch: `main`
- Application entry point: `app.py`
- Model checkpoint: `artifacts/model_expanded_v2_candidate.pt`

## Limitations and Future Improvements

The current model achieves 82.03% test accuracy, leaving room for improvement.

Potential improvements include:

- Reducing confusion between visually similar breeds.
- Improving performance on underperforming breed categories.
- Reducing the training-validation performance gap.
- Evaluating alternative architectures.
- Applying hyperparameter optimization.
- Exploring model calibration and inference optimization.
- Adding Grad-CAM visualizations for model interpretability.

## Research Notes

The model's softmax confidence score should not be interpreted as a calibrated probability.

The reported test metrics are from the project's existing evaluation. Because the test set was also used in model comparisons, an independent external evaluation would provide stronger evidence of generalization.

## License and Dataset

The project uses the Stanford Dogs Dataset for academic and research purposes.

Review the dataset's original usage and redistribution terms before redistributing images or dataset files.

---

**Developed with Python, PyTorch, EfficientNet-B0, and Streamlit.**# Stanford Dogs Breed Classifier

<escape>🚀 **Live Demo:** [Try the Stanford Dogs Classifier](https://petclassifier-zgg5u5pnnfprkygz6ipu8f.streamlit.app/)</escape>

A deep-learning image classification application that identifies 120 dog breeds using EfficientNet-B0 transfer learning.

## Model Performance

- Test Accuracy: 82.03%
- Macro F1-Score: 81.56%
- Number of Classes: 120

## Technology Stack

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- Streamlit
- Stanford Dogs Dataset

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Live Application

https://petclassifier-zgg5u5pnnfprkygz6ipu8f.streamlit.app/# Pet Breed Classifier

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
