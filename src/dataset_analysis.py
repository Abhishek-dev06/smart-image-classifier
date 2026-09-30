"""Analyze the dataset: class counts, image sizes and class imbalance."""

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up window needed
import matplotlib.pyplot as plt
from PIL import Image

from utils import (RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, RESULTS_DIR,
                   VALID_EXTENSIONS, get_class_names)


def list_images(class_dir):
    return [p for p in class_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS]


def count_per_class(root, class_names):
    """Return {class_name: number_of_images} for a dataset folder."""
    return {name: len(list_images(root / name)) if (root / name).exists() else 0
            for name in class_names}


def image_size_stats(root, class_names):
    """Read image dimensions (only headers are read, so this is fast)."""
    sizes = Counter()
    widths, heights = [], []
    for name in class_names:
        for path in list_images(root / name):
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:
                continue
            sizes[(w, h)] += 1
            widths.append(w)
            heights.append(h)
    return sizes, widths, heights


def describe_imbalance(counts):
    biggest = max(counts, key=counts.get)
    smallest = min(counts, key=counts.get)
    ratio = counts[biggest] / counts[smallest]
    if ratio < 1.5:
        verdict = "The dataset is fairly balanced."
    elif ratio < 3:
        verdict = "The dataset is moderately imbalanced."
    else:
        verdict = "The dataset is significantly imbalanced."
    return biggest, smallest, ratio, verdict


def plot_distribution(counts, save_path):
    names = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color="steelblue")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value),
                 ha="center", va="bottom")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    class_names = get_class_names(RAW_DIR)
    counts = count_per_class(RAW_DIR, class_names)
    total = sum(counts.values())

    print("=" * 40)
    print("DATASET SUMMARY")
    print("=" * 40)
    print(f"Total images: {total}")
    print(f"Classes: {len(class_names)}\n")
    for name, n in counts.items():
        print(f"{name:<12} {n:>5}  ({n / total * 100:.1f}%)")

    # Image dimensions
    sizes, widths, heights = image_size_stats(RAW_DIR, class_names)
    print("\nIMAGE DIMENSIONS")
    print(f"Unique sizes: {len(sizes)}")
    print("Most common sizes (width x height):")
    for (w, h), n in sizes.most_common(3):
        print(f"  {w} x {h}: {n} images")
    print(f"Width  min/max: {min(widths)} / {max(widths)}")
    print(f"Height min/max: {min(heights)} / {max(heights)}")

    # Imbalance
    biggest, smallest, ratio, verdict = describe_imbalance(counts)
    print("\nCLASS BALANCE")
    print(f"Largest class : {biggest} ({counts[biggest]})")
    print(f"Smallest class: {smallest} ({counts[smallest]})")
    print(f"Ratio (largest/smallest): {ratio:.2f}")
    print(verdict)

    # Split counts
    print("\nSPLIT COUNTS")
    split_summary = {}
    for split_name, folder in (("train", TRAIN_DIR),
                               ("validation", VAL_DIR),
                               ("test", TEST_DIR)):
        if folder.exists():
            split_counts = count_per_class(folder, class_names)
            split_summary[split_name] = split_counts
            print(f"{split_name:<11} {sum(split_counts.values())}")
        else:
            print(f"{split_name:<11} missing (run prepare_dataset.py first)")

    # Save outputs
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_distribution(counts, RESULTS_DIR / "class_distribution.png")
    summary = {
        "total_images": total,
        "num_classes": len(class_names),
        "class_counts": counts,
        "imbalance_ratio": round(ratio, 2),
        "verdict": verdict,
        "splits": split_summary,
    }
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved: results/class_distribution.png")
    print("Saved: results/dataset_summary.json")


if __name__ == "__main__":
    main()