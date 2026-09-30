"""Train the baseline CNN and save the best model, class names and plots."""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from tensorflow import keras

from data_pipeline import load_datasets
from model import build_model, count_parameters
from utils import (MODEL_PATH, CLASS_NAMES_PATH, MODELS_DIR, RESULTS_DIR,
                   set_seed)


def build_callbacks():
    return [
        # Stop when validation loss stops improving; go back to best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        # Halve the learning rate when progress stalls
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        # Keep the best model on disk
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=1),
    ]


def save_history_plots(history):
    """Save training vs validation accuracy and loss plots."""
    epochs = range(1, len(history["accuracy"]) + 1)

    for metric, title, filename in (
        ("accuracy", "Training vs Validation Accuracy", "training_accuracy.png"),
        ("loss", "Training vs Validation Loss", "training_loss.png"),
    ):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history[metric], marker="o", label="Training")
        plt.plot(epochs, history[f"val_{metric}"], marker="o", label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the image classifier.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal flip (use for traffic signs).")
    args = parser.parse_args()

    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, _, class_names = load_datasets()  # test set is NOT used here
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"\nClasses: {class_names}")

    model = build_model(len(class_names), allow_horizontal_flip=not args.no_flip)
    total, trainable, _ = count_parameters(model)
    print(f"Model parameters: {total:,} (trainable: {trainable:,})\n")

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=build_callbacks(),
    ).history
    training_seconds = time.time() - start

    # EarlyStopping restored the best weights, so this saves the best model.
    model.save(MODEL_PATH)

    save_history_plots(history)
    best_epoch = min(range(len(history["val_loss"])),
                     key=lambda i: history["val_loss"][i])
    summary = {
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_loss": round(history["val_loss"][best_epoch], 4),
        "val_accuracy_at_best_epoch": round(history["val_accuracy"][best_epoch], 4),
        "train_accuracy_at_best_epoch": round(history["accuracy"][best_epoch], 4),
        "training_time_seconds": round(training_seconds, 1),
        "total_parameters": total,
        "class_names": class_names,
    }
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()},
                  f, indent=2)

    print("\n" + "=" * 40)
    print("TRAINING FINISHED")
    print("=" * 40)
    print(f"Epochs run             : {summary['epochs_run']}")
    print(f"Best epoch             : {summary['best_epoch']}")
    print(f"Train accuracy (best)  : {summary['train_accuracy_at_best_epoch'] * 100:.2f}%")
    print(f"Validation accuracy    : {summary['val_accuracy_at_best_epoch'] * 100:.2f}%")
    print(f"Training time          : {training_seconds / 60:.1f} minutes")
    print(f"Model saved to         : models/{MODEL_PATH.name}")
    print(f"Class names saved to   : models/{CLASS_NAMES_PATH.name}")


if __name__ == "__main__":
    main()