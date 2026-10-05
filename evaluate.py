import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path", default="models/recyclevision.keras")
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = tf.keras.utils.image_dataset_from_directory(args.data_dir, validation_split=0.2, subset="validation", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    labels = ds.class_names
    model = tf.keras.models.load_model(args.model_path)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=180)
    print(json.dumps(report["macro avg"], indent=2))


if __name__ == "__main__":
    main()
