"""Model definitions (WasteWealth MobileNetV2 and baseline CNN) for Smart Image Classifier."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from data_pipeline import build_augmentation, build_preprocessing
from utils import IMAGE_SIZE, RESULTS_DIR, SEED, set_seed


def build_mobilenet_v2_model(num_classes=12, allow_horizontal_flip=True,
                             freeze_backbone=True, pretrained=True):
    """Build and compile the WasteWealth MobileNetV2 classifier.

    Augmentation and normalization are part of the model.
    pretrained=False skips the ImageNet weight download (useful for offline tests).
    """
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")

    x = build_augmentation(allow_horizontal_flip)(inputs)
    x = build_preprocessing()(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    if freeze_backbone:
        base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, seed=SEED, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="WasteWealth_MobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn_model(num_classes=12, allow_horizontal_flip=True):
    """Build and compile the baseline CNN."""
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


def build_model(num_classes=12, allow_horizontal_flip=True, backbone="mobilenet_v2"):
    """Build and compile the classifier.

    Supports 'mobilenet_v2' (WasteWealth MobileNetV2, default) and 'baseline_cnn'.
    """
    if backbone == "baseline_cnn":
        return build_baseline_cnn_model(num_classes, allow_horizontal_flip=allow_horizontal_flip)
    return build_mobilenet_v2_model(num_classes, allow_horizontal_flip=allow_horizontal_flip)


def count_parameters(model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return trainable + non_trainable, trainable, non_trainable


def main():
    set_seed()
    num_classes = 12  # WasteWealth MobileNetV2 12-class dataset
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