import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, callbacks, applications, Model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def build_model(num_classes: int) -> Model:
    base = applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(len(class_names))
    stop = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2)
    checkpoint = callbacks.ModelCheckpoint(out / "recyclevision.keras", monitor="val_accuracy", save_best_only=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[stop, reduce, checkpoint])
    with open(out / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved model and labels to {out}")


if __name__ == "__main__":
    main()
