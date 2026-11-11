from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model_expanded_v2_candidate.pt"
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Stanford Dogs Classifier",
    page_icon="🐕",
    layout="centered",
)

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    classes = checkpoint["classes"]
    model = models.efficientnet_b0(weights=None)

    old_head = model.classifier[-1]
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(old_head.in_features, len(classes)),
    )

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    return model, classes, checkpoint

st.title("🐕 Stanford Dogs Breed Classifier")

st.write(
    "Identify dog breeds using a deep learning model "
    "trained on 120 Stanford Dogs categories."
)

model, classes, checkpoint = load_model()

img_size = int(checkpoint["img_size"])

transform = transforms.Compose([
    transforms.Resize(int(img_size * 1.14)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

uploaded_file = st.file_uploader(
    "Upload a dog photograph",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded photograph",
                 use_container_width=True)

        tensor = transform(image).unsqueeze(0)

        with torch.inference_mode():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        values, indices = torch.topk(probabilities, 3)

        st.subheader("Prediction")

        breed = classes[indices[0].item()].replace("_", " ").title()
        confidence = values[0].item() * 100

        st.success(f"Predicted breed: {breed}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.subheader("Top 3 predictions")

        for rank, (value, index) in enumerate(
            zip(values.tolist(), indices.tolist()), start=1
        ):
            name = classes[index].replace("_", " ").title()
            st.write(f"**{rank}. {name} — {value * 100:.2f}%**")
            st.progress(value)

        st.caption(
            "Confidence represents the model's softmax score, "
            "not a guarantee of prediction correctness."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")

st.divider()
st.caption("EfficientNet-B0 | 120 dog breeds | Research model V2")
