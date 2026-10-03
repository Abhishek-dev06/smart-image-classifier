"""Confusion matrix and most-confused class pairs from saved test predictions."""

import json

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from utils import CLASS_NAMES_PATH, PROJECT_ROOT, RESULTS_DIR


def load_inputs():
    predictions_path = RESULTS_DIR / "test_predictions.json"
    if not predictions_path.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Missing test_predictions.json or class_names.json.\n"
            "Run: python src/train.py  and then  python src/evaluate.py")
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)
    return class_names, predictions


def draw_matrix(ax, matrix, class_names, title, fmt):
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")


def top_confusions(matrix, class_names, limit=5):
    """Largest off-diagonal cells: (actual, predicted, count, % of actual class)."""
    pairs = []
    for i, actual in enumerate(class_names):
        row_total = matrix[i].sum()
        for j, predicted in enumerate(class_names):
            if i != j and matrix[i, j] > 0:
                pairs.append({
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(matrix[i, j]),
                    "percent_of_actual_class": round(
                        float(matrix[i, j] / row_total * 100), 1),
                })
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:limit]


def save_example_grid(predictions, actual, predicted, save_path, max_images=8):
    """Save a grid of test images of class `actual` that were predicted as `predicted`."""
    wrong = [p for p in predictions
             if p["actual"] == actual and p["predicted"] == predicted][:max_images]
    if not wrong:
        return
    cols = min(4, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    plt.figure(figsize=(3 * cols, 3.2 * rows))
    for k, item in enumerate(wrong, start=1):
        plt.subplot(rows, cols, k)
        plt.imshow(Image.open(PROJECT_ROOT / item["file"]).convert("RGB"))
        plt.title(f"{item['confidence'] * 100:.0f}% confident", fontsize=9)
        plt.axis("off")
    plt.suptitle(f"Actual: {actual}  ->  Predicted: {predicted}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    class_names, predictions = load_inputs()
    index = {name: i for i, name in enumerate(class_names)}
    y_true = np.array([index[p["actual"]] for p in predictions])
    y_pred = np.array([index[p["predicted"]] for p in predictions])

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)   # each row = recall per class

    # Two panels: raw counts and row-normalized percentages
    size = max(7, len(class_names) * 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(size * 2, size))
    draw_matrix(axes[0], matrix, class_names, "Confusion Matrix (counts)", "d")
    draw_matrix(axes[1], normalized, class_names,
                "Confusion Matrix (row-normalized)", ".2f")
    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    accuracy = np.trace(matrix) / matrix.sum()
    print(f"Test images: {matrix.sum()}")
    print(f"Accuracy from matrix (diagonal / total): {accuracy * 100:.2f}%")
    print("Per-class recall (row-normalized diagonal):")
    for name, value in zip(class_names, np.diag(normalized)):
        print(f"  {name:<12} {value * 100:.1f}%")

    pairs = top_confusions(matrix, class_names)
    print("\nMost common confusions (actual -> predicted):")
    if not pairs:
        print("  None. Every test image was classified correctly.")
    for p in pairs:
        print(f"  {p['actual']} -> {p['predicted']}: {p['count']} images "
              f"({p['percent_of_actual_class']}% of {p['actual']})")

    with open(RESULTS_DIR / "confusion_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    if pairs:
        top = pairs[0]
        save_example_grid(predictions, top["actual"], top["predicted"],
                          RESULTS_DIR / "top_confusion_examples.png")
        print("\nSaved: results/top_confusion_examples.png")

    print("Saved: results/confusion_matrix.png")
    print("Saved: results/confusion_pairs.json")


if __name__ == "__main__":
    main()