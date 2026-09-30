"""Print a facts sheet (real numbers) to use while writing DECISIONS.md.

Usage: python src/decisions_facts.py
"""

import json
import sys

from utils import RESULTS_DIR


def read(name, required=True):
    path = RESULTS_DIR / name
    if not path.exists():
        if required:
            sys.exit(f"Missing results/{name}. Run the earlier steps first.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    dataset = read("dataset_summary.json")
    training = read("training_summary.json")
    history = read("training_history.json")
    metrics = read("evaluation_metrics.json")
    pairs = read("confusion_pairs.json", required=False) or []
    errors = read("error_analysis.json", required=False)
    unseen = read("unseen_predictions.json", required=False) or []

    splits = dataset.get("splits", {})
    sizes = {k: sum(v.values()) for k, v in splits.items()}

    print("=" * 46)
    print("FACTS SHEET FOR DECISIONS.md")
    print("=" * 46)

    print("\n[Dataset]")
    print(f"Total images : {dataset['total_images']}")
    print(f"Classes      : {dataset['num_classes']}")
    for name, count in dataset["class_counts"].items():
        print(f"  {name:<12} {count}")
    print(f"Imbalance ratio (largest/smallest): {dataset['imbalance_ratio']}")
    print(f"Verdict: {dataset['verdict']}")

    print("\n[Split]")
    total = sum(sizes.values()) or 1
    for split_name, count in sizes.items():
        print(f"  {split_name:<11} {count} ({count / total * 100:.1f}%)")

    print("\n[Training]")
    best = training["best_epoch"] - 1
    train_acc = history["accuracy"][best] * 100
    val_acc = history["val_accuracy"][best] * 100
    print(f"Epochs run: {training['epochs_run']} (best epoch {training['best_epoch']})")
    print(f"Train accuracy at best epoch     : {train_acc:.2f}%")
    print(f"Validation accuracy at best epoch: {val_acc:.2f}%")
    print(f"Gap (train - validation)         : {train_acc - val_acc:+.2f} points")
    print("Note: training accuracy is measured with augmentation switched on,")
    print("so it can look lower than it would on clean training images.")
    print(f"Training time: {training['training_time_seconds'] / 60:.1f} min")
    print(f"Parameters   : {training['total_parameters']:,}")

    print("\n[Test set]")
    print(f"Test images  : {metrics['num_test_images']}")
    print(f"Test accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1     : {metrics['macro_f1']:.3f}")
    print(f"Weakest class by F1: {metrics['weakest_class_by_f1']}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:<12} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (n={m['support']})")

    print("\n[Most common confusions]")
    if not pairs:
        print("  none")
    for p in pairs[:3]:
        print(f"  {p['actual']} -> {p['predicted']}: {p['count']} "
              f"({p['percent_of_actual_class']}% of {p['actual']})")

    if unseen:
        print("\n[Unseen images]")
        for u in unseen:
            print(f"  {u['file'].split('/')[-1]:<28} expected={u.get('expected')} "
                  f"predicted={u['predicted']} ({u['confidence'] * 100:.1f}%)")

    if errors:
        f = errors["featured_failure"]
        print("\n[Featured failure]")
        print(f"Image     : {f['file']}")
        print(f"Source    : {f['source']}")
        print(f"Actual    : {f['actual']}")
        print(f"Predicted : {f['predicted']}")
        print(f"Confidence: {f['confidence'] * 100:.2f}%")
        train = dataset.get("splits", {}).get("train", {})
        if train:
            print(f"Training images: {f['actual']}={train.get(f['actual'])}, "
                  f"{f['predicted']}={train.get(f['predicted'])}")
    else:
        print("\n[Featured failure] none saved. Run: python src/error_analysis.py")


if __name__ == "__main__":
    main()