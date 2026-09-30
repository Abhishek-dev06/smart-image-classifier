"""Build the README 'Results' section from the real saved result files.

Usage: python src/generate_results_section.py
Output: results/results_section.md  (paste it into README.md)
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
    metrics = read("evaluation_metrics.json")
    pairs = read("confusion_pairs.json", required=False) or []
    unseen = read("unseen_predictions.json", required=False) or []

    splits = dataset.get("splits", {})
    n = {k: sum(v.values()) for k, v in splits.items()}

    lines = ["## Results", ""]
    lines += ["All numbers below were produced by running the scripts in this "
              "repository. The test set was used only once, for final evaluation.", ""]

    lines += ["### Dataset split", "",
              "| Split | Images |", "|---|---|",
              f"| Training | {n.get('train', '?')} |",
              f"| Validation | {n.get('validation', '?')} |",
              f"| Test | {n.get('test', '?')} |", ""]

    lines += ["### Summary", "",
              "| Metric | Value |", "|---|---|",
              f"| Best epoch | {training['best_epoch']} of {training['epochs_run']} |",
              f"| Validation accuracy (best epoch) | {training['val_accuracy_at_best_epoch'] * 100:.2f}% |",
              f"| Test accuracy | {metrics['test_accuracy'] * 100:.2f}% |",
              f"| Test loss | {metrics['test_loss']:.4f} |",
              f"| Macro precision | {metrics['macro_precision']:.3f} |",
              f"| Macro recall | {metrics['macro_recall']:.3f} |",
              f"| Macro F1 | {metrics['macro_f1']:.3f} |",
              f"| Training time | {training['training_time_seconds'] / 60:.1f} min (CPU) |",
              f"| Parameters | {training['total_parameters']:,} |", ""]

    lines += ["### Per-class performance", "",
              "| Class | Precision | Recall | F1 | Test images |", "|---|---|---|---|---|"]
    for name, m in metrics["per_class"].items():
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1']:.3f} | {m['support']} |")
    lines += ["", f"Weakest class by F1: **{metrics['weakest_class_by_f1']}**.", ""]

    lines += ["### Confusion matrix", "",
              "![Confusion matrix](results/confusion_matrix.png)", ""]
    if pairs:
        lines += ["Most common confusions (actual -> predicted):", ""]
        for p in pairs[:3]:
            lines.append(f"- {p['actual']} -> {p['predicted']}: {p['count']} images "
                         f"({p['percent_of_actual_class']}% of {p['actual']})")
        lines.append("")

    lines += ["### Training curves", "",
              "![Accuracy](results/training_accuracy.png)",
              "![Loss](results/training_loss.png)", ""]

    if unseen:
        lines += ["### Unseen image tests", "",
                  "| Image | Expected | Predicted | Confidence |", "|---|---|---|---|"]
        for u in unseen:
            lines.append(f"| {u['file'].split('/')[-1]} | {u.get('expected') or '-'} | "
                         f"{u['predicted']} | {u['confidence'] * 100:.2f}% |")
        lines.append("")

    out = RESULTS_DIR / "results_section.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: results/{out.name}")


if __name__ == "__main__":
    main()