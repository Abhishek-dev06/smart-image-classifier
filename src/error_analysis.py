"""Find misclassified images and document one failure case.

Usage:
    python src/error_analysis.py            # feature the most confident mistake
    python src/error_analysis.py --pick 3   # feature the 3rd item in the list
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predict import load_model_and_classes, load_image, predict
from utils import PROJECT_ROOT, RESULTS_DIR


def read_json(name, required=False):
    path = RESULTS_DIR / name
    if not path.exists():
        if required:
            sys.exit(f"Error: {path.name} not found. Run: python src/evaluate.py")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_failures(test_predictions, unseen_predictions):
    """Combine wrong predictions from the test set and unseen images."""
    failures = []
    for item in test_predictions:
        if not item["correct"]:
            failures.append({"source": "test", "file": item["file"],
                             "actual": item["actual"],
                             "predicted": item["predicted"],
                             "confidence": item["confidence"]})
    for item in unseen_predictions:
        if item.get("correct") is False:
            failures.append({"source": "unseen", "file": item["file"],
                             "actual": item["expected"],
                             "predicted": item["predicted"],
                             "confidence": item["confidence"]})
    # Most confident mistakes first: they are the most informative failures
    failures.sort(key=lambda f: f["confidence"], reverse=True)
    return failures


def save_failure_grid(failures, save_path, max_images=12):
    shown = [f for f in failures if f["source"] == "test"][:max_images]
    if not shown:
        return False
    cols = min(4, len(shown))
    rows = int(np.ceil(len(shown) / cols))
    plt.figure(figsize=(3.2 * cols, 3.5 * rows))
    for k, item in enumerate(shown, start=1):
        plt.subplot(rows, cols, k)
        plt.imshow(Image.open(PROJECT_ROOT / item["file"]).convert("RGB"))
        plt.title(f"actual: {item['actual']}\npredicted: {item['predicted']} "
                  f"({item['confidence'] * 100:.0f}%)", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def save_failure_case(model, class_names, failure, save_path):
    """Image on the left, top-5 probabilities on the right."""
    image = load_image(PROJECT_ROOT / failure["file"])
    ranked = predict(model, class_names, image)[:5]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(f"{failure['file'].split('/')[-1]}\n"
                     f"Actual: {failure['actual']}", fontsize=10)

    names = [n for n, _ in ranked][::-1]
    values = [p * 100 for _, p in ranked][::-1]
    colors = ["tomato" if n == failure["predicted"] else "steelblue" for n in names]
    ax_bar.barh(names, values, color=colors)
    for y, value in enumerate(values):
        ax_bar.text(value + 1, y, f"{value:.1f}%", va="center")
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Softmax probability (%)")
    ax_bar.set_title(f"Predicted: {failure['predicted']}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return ranked


def train_counts():
    """Training images per class (from Step 4), if available."""
    path = RESULTS_DIR / "dataset_summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("splits", {}).get("train", {})


def main():
    parser = argparse.ArgumentParser(description="Analyze misclassified images.")
    parser.add_argument("--pick", type=int, default=1,
                        help="which failure to feature (1 = most confident mistake)")
    args = parser.parse_args()

    test_predictions = read_json("test_predictions.json", required=True)
    unseen_predictions = read_json("unseen_predictions.json")
    failures = collect_failures(test_predictions, unseen_predictions)

    test_wrong = [p for p in test_predictions if not p["correct"]]
    test_right = [p for p in test_predictions if p["correct"]]
    print(f"Test images: {len(test_predictions)} | wrong: {len(test_wrong)}")
    if test_right:
        print(f"Average confidence when correct: "
              f"{np.mean([p['confidence'] for p in test_right]) * 100:.1f}%")
    if test_wrong:
        print(f"Average confidence when wrong  : "
              f"{np.mean([p['confidence'] for p in test_wrong]) * 100:.1f}%")

    if not failures:
        print("\nNo misclassified image was found in the test set or in the unseen images.")
        print("Do NOT invent a failure. Add harder unseen images to test_images/")
        print("(transparent plastic, bad lighting, cluttered background, partly hidden")
        print("objects) and run: python src/predict.py --save   then this script again.")
        return

    print(f"\nMisclassified images (most confident first): {len(failures)}")
    for rank, f in enumerate(failures[:15], start=1):
        print(f"{rank:>3}. [{f['source']}] {f['file']}\n"
              f"     actual={f['actual']}  predicted={f['predicted']}  "
              f"confidence={f['confidence'] * 100:.1f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    if save_failure_grid(failures, RESULTS_DIR / "misclassified_examples.png"):
        print("\nSaved: results/misclassified_examples.png")

    index = min(max(args.pick, 1), len(failures)) - 1
    featured = failures[index]
    model, class_names = load_model_and_classes()
    ranked = save_failure_case(model, class_names, featured,
                               RESULTS_DIR / "failure_case.png")

    counts = train_counts()
    print("\n" + "=" * 44)
    print("FEATURED FAILURE (copy into DECISIONS.md)")
    print("=" * 44)
    print(f"Image     : {featured['file']}")
    print(f"Source    : {featured['source']} image")
    print(f"Actual    : {featured['actual']}")
    print(f"Predicted : {featured['predicted']}")
    print(f"Confidence: {featured['confidence'] * 100:.2f}%")
    print("Top predictions: " + ", ".join(f"{n} {p * 100:.1f}%" for n, p in ranked[:3]))
    if counts:
        print(f"Training images: {featured['actual']}={counts.get(featured['actual'])}, "
              f"{featured['predicted']}={counts.get(featured['predicted'])}")

    with open(RESULTS_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"total_failures": len(failures),
                   "featured_failure": featured,
                   "all_failures": failures}, f, indent=2)
    print("\nSaved: results/failure_case.png")
    print("Saved: results/error_analysis.json")


if __name__ == "__main__":
    main()