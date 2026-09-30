"""Data loading, preprocessing and augmentation for Smart Image Classifier."""

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import (TRAIN_DIR, VAL_DIR, TEST_DIR, RESULTS_DIR,
                   IMAGE_SIZE, BATCH_SIZE, SEED, set_seed)


def _check_folder(folder):
    if not folder.exists() or not any(folder.iterdir()):
        raise FileNotFoundError(
            f"Dataset folder missing or empty: {folder}\n"
            "Run: python src/prepare_dataset.py"
        )


def _load_split(folder, shuffle):
    """Load one split. Images are resized to IMAGE_SIZE and converted to RGB."""
    _check_folder(folder)
    return keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",      # integer labels -> SparseCategoricalCrossentropy
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def load_datasets():
    """Return (train_ds, val_ds, test_ds, class_names)."""
    # Only training data is shuffled. Test must stay in fixed order so that
    # predictions line up with labels in the confusion matrix.
    train_ds = _load_split(TRAIN_DIR, shuffle=True)
    val_ds = _load_split(VAL_DIR, shuffle=False)
    test_ds = _load_split(TEST_DIR, shuffle=False)

    class_names = train_ds.class_names
    if not (class_names == val_ds.class_names == test_ds.class_names):
        raise ValueError("Class folders differ between train/validation/test.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_augmentation(allow_horizontal_flip=True):
    """Augmentation layers. Active only during training (training=True).

    Set allow_horizontal_flip=False for datasets where flipping changes the
    meaning of the class (for example traffic signs with left/right arrows).
    """
    aug_layers = []
    if allow_horizontal_flip:
        aug_layers.append(layers.RandomFlip("horizontal", seed=SEED))
    aug_layers.append(layers.RandomRotation(0.1, seed=SEED))  # about +-36 degrees
    aug_layers.append(layers.RandomZoom(0.1, seed=SEED))      # +-10% zoom
    return keras.Sequential(aug_layers, name="data_augmentation")


def build_preprocessing():
    """Normalize pixel values from [0, 255] to [0, 1]."""
    return layers.Rescaling(1.0 / 255, name="normalization")


def save_augmentation_preview(train_ds, class_names, save_path):
    """Save a figure: one original image and 7 augmented versions of it."""
    augmentation = build_augmentation()
    images, labels = next(iter(train_ds))
    image = images[0]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 4, 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(f"Original\n({class_names[int(labels[0])]})")
    plt.axis("off")

    for i in range(2, 9):
        augmented = augmentation(tf.expand_dims(image, 0), training=True)[0]
        augmented = tf.clip_by_value(augmented, 0, 255)
        plt.subplot(2, 4, i)
        plt.imshow(augmented.numpy().astype("uint8"))
        plt.title("Augmented")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    set_seed()
    train_ds, val_ds, test_ds, class_names = load_datasets()

    print(f"\nClasses ({len(class_names)}): {class_names}")
    print(f"Train batches     : {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches      : {tf.data.experimental.cardinality(test_ds).numpy()}")

    images, labels = next(iter(train_ds))
    print(f"\nOne batch -> images: {images.shape}, labels: {labels.shape}")
    print(f"Pixel range before normalization: "
          f"{float(tf.reduce_min(images)):.0f} to {float(tf.reduce_max(images)):.0f}")

    normalized = build_preprocessing()(images)
    print(f"Pixel range after normalization : "
          f"{float(tf.reduce_min(normalized)):.2f} to {float(tf.reduce_max(normalized)):.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    save_path = RESULTS_DIR / "augmentation_samples.png"
    save_augmentation_preview(train_ds, class_names, save_path)
    print(f"\nSaved: {save_path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()