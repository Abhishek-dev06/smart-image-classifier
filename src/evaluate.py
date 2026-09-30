"""Evaluate the trained model on the TEST set only."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras

from data_pipeline import load_test_dataset_with_paths
from utils import (MODEL_PATH, CLASS_NAMES_PATH, RESULTS_DIR, PROJECT_ROOT,
                   set_seed)


def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun: python src/train.py")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}\nRun: python src/train.py")
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def relative_path(path):
    """Store paths relative to the project root so they work on any computer."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def main():
    set_seed()
    model, class_names = load_model_and_classes()
    test_ds, ds_class_names, file_paths = load_test_dataset_with_paths()

    if ds_class_names != class_names:
        raise ValueError("Class names in the test folder do not match the model.")

    # True labels (test order is fixed because shuffle=False)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    # Sanity check: sklearn and Keras must agree
    assert abs(accuracy_score(y_true, y_pred) - test_accuracy) < 1e-4

    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0)

    print("\n" + classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0))

    per_class_f1 = {name: report[name]["f1-score"] for name in class_names}
    weakest = min(per_class_f1, key=per_class_f1.get)

    metrics = {
        "num_test_images": int(len(y_true)),
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "weakest_class_by_f1": weakest,
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }

    # Predictions are reused for the confusion matrix and error analysis
    predictions = [
        {
            "file": relative_path(path),
            "actual": class_names[int(t)],
            "predicted": class_names[int(p)],
            "confidence": round(float(c), 4),
            "correct": bool(t == p),
        }
        for path, t, p, c in zip(file_paths, y_true, y_pred, confidence)
    ]

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    wrong = sum(not p["correct"] for p in predictions)
    print("=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    print(f"Test images    : {metrics['num_test_images']}")
    print(f"Test loss      : {metrics['test_loss']:.4f}")
    print(f"Test accuracy  : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Macro precision: {metrics['macro_precision']:.3f}")
    print(f"Macro recall   : {metrics['macro_recall']:.3f}")
    print(f"Macro F1       : {metrics['macro_f1']:.3f}")
    print(f"Weakest class  : {weakest} (F1 {per_class_f1[weakest]:.3f})")
    print(f"Wrong predictions: {wrong} of {len(predictions)}")
    print("\nSaved: results/evaluation_metrics.json")
    print("Saved: results/test_predictions.json")


if __name__ == "__main__":
    main()