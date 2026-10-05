from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)


def load_artifacts(model_path="models/recyclevision.keras", labels_path="models/class_names.json"):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels


def predict_image(image: Image.Image, model, labels, top_k=3):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)[None, ...]
    probs = model.predict(arr, verbose=0)[0]
    indices = np.argsort(probs)[::-1][:top_k]
    return [(labels[int(i)], float(probs[int(i)])) for i in indices]
