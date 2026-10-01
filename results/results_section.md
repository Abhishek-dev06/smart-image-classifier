## Results

These numbers come from the saved result files in this repository. They are historical results, not a fresh evaluation. The source dataset and its provenance must be restored to reproduce them.

### Dataset split

| Split | Images |
|---|---|
| Training | 3252 |
| Validation | 696 |
| Test | 702 |

### Summary

| Metric | Value |
|---|---|
| Best epoch | 30 of 30 |
| Validation accuracy (best epoch) | 69.54% |
| Test accuracy | 67.95% |
| Test loss | 0.7829 |
| Macro precision | 0.698 |
| Macro recall | 0.679 |
| Macro F1 | 0.676 |
| Recorded training time | 36.4 min |
| Parameters | 110,534 |

### Per-class performance

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| battery | 0.832 | 0.803 | 0.817 | 117 |
| glass | 0.711 | 0.504 | 0.590 | 117 |
| metal | 0.555 | 0.607 | 0.580 | 117 |
| organic | 0.839 | 0.803 | 0.821 | 117 |
| paper | 0.557 | 0.880 | 0.682 | 117 |
| plastic | 0.691 | 0.479 | 0.566 | 117 |

Weakest class by F1: **plastic**.

### Confusion matrix

![Confusion matrix](results/confusion_matrix.png)

Most common confusions (actual -> predicted):

- metal -> paper: 26 images (22.2% of metal)
- plastic -> paper: 23 images (19.7% of plastic)
- glass -> metal: 20 images (17.1% of glass)

### Training curves

![Accuracy](results/training_accuracy.png)
![Loss](results/training_loss.png)

### Unseen image tests

| Image | Expected | Predicted | Confidence |
|---|---|---|---|
| class_car.jpg | - | glass | 98.40% |
| class_favicon.jpeg | - | paper | 48.15% |
| class_pihu.png | - | organic | 63.70% |
| class_plush.png | - | organic | 97.72% |
| class_sshot.png | - | battery | 78.98% |
