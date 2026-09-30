"""Baseline CNN for Smart Image Classifier."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from data_pipeline import build_augmentation, build_preprocessing
from utils import IMAGE_SIZE, RESULTS_DIR, SEED, set_seed


def build_model(num_classes, allow_horizontal_flip=True):
    """Build and compile the baseline CNN.

    Augmentation and normalization are part of the model, so predict.py and
    the Streamlit app only need to resize the image. Augmentation layers are
    active only during training.
    """
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")

    x = build_augmentation(allow_horizontal_flip)(inputs)
    x = build_preprocessing()(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, kernel_size=3, padding="same",
                          activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3, seed=SEED)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def count_parameters(model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return trainable + non_trainable, trainable, non_trainable


def main():
    set_seed()
    num_classes = 6  # change this if your dataset has a different class count
    model = build_model(num_classes)
    model.summary()

    total, trainable, non_trainable = count_parameters(model)
    print(f"\nTotal parameters        : {total:,}")
    print(f"Trainable parameters    : {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")

    # Sanity check: random batch -> output shape and softmax sum
    dummy = np.random.randint(0, 256, size=(4, *IMAGE_SIZE, 3)).astype("float32")
    probs = model.predict(dummy, verbose=0)
    print(f"\nOutput shape: {probs.shape}")
    print(f"Each row sums to 1: {np.allclose(probs.sum(axis=1), 1.0)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    print("Saved: results/model_summary.txt")


if __name__ == "__main__":
    main()