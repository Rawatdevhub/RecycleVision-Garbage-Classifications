from pathlib import Path
import streamlit as st
from PIL import Image
from src.predict import load_artifacts, predict_image

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")
st.title("♻️ RecycleVision")
st.caption("Garbage image classification with transfer learning")

MODEL_PATH = Path("recyclevision.keras")
LABELS_PATH = Path("class_names.json")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("Model files are missing. Run train.py first and place the outputs in models/.")
    st.stop()

@st.cache_resource
def get_model():
    return load_artifacts(MODEL_PATH, LABELS_PATH)

model, labels = get_model()
file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    predictions = predict_image(image, model, labels, top_k=3)
    top_label, top_score = predictions[0]
    st.subheader(f"Prediction: {top_label.title()}")
    st.metric("Confidence", f"{top_score:.1%}")
    st.write("Top alternatives")
    for label, score in predictions:
        st.progress(score, text=f"{label.title()} — {score:.1%}")
    if top_score < 0.60:
        st.warning("Low-confidence result: review the image and local recycling guidance.")

st.divider()
st.caption("Prototype only. Predictions are decision support and may be affected by lighting, clutter, and ambiguous waste labels.")
