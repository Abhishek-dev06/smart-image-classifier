# Engineering Decisions

This document records the architecture, the evidence available in the repository, and decisions made while repairing photo inference. Recorded experiment results are distinguished from verified software behavior. The project remains a six-category waste classifier.

## 1. Keep the model's intended scope

**Decision:** retain the bundled waste model and the ordered labels `battery`, `glass`, `metal`, `organic`, `paper`, and `plastic`.

**Reason:** the model has six outputs trained for material categories. Renaming labels or changing preprocessing cannot turn it into a general photo recognizer. The maintainer requested that waste classification remain the project's purpose.

**Consequence:** the UI and CLI state the scope before displaying predictions. All inputs still receive a ranked list over the six classes. No automatic out-of-distribution rejection is claimed. The 0.60 warning threshold is a heuristic, not a calibrated acceptance threshold.

## 2. Dataset evidence and reproducibility

The committed summary records 4,650 images: 775 per class, with 542 training, 116 validation, and 117 test images per class. The resulting totals are 3,252 / 696 / 702. The imbalance ratio is 1.0, so class weighting is not a priority for this recorded baseline.

The dataset name, source URL, license, filtering procedure, and split manifest are missing. The raw images and prepared splits are not committed. Consequently, this checkout supports inference and software testing but cannot independently reproduce the original training or test evaluation. No specific dataset identity or use history is asserted.

**Decision:** describe these gaps explicitly instead of inventing dataset attribution or claiming the test set was evaluated exactly once.

## 3. Split before augmentation and validate before replacement

The existing split procedure operates per class with seed 42 and ratios 70% / 15% / remaining 15%. Validation loss controls checkpoint selection and learning-rate adjustment; the test split is intended for final evaluation. Augmentation is inside the model and inactive during inference.

**Repair:** validate every raw class before removing existing split directories. Previously, a later class with fewer than ten valid images could fail after existing splits had already been deleted. A regression test preserves a sentinel from an existing split when validation fails.

**Limitations:** splitting remains file-based, not grouped by subject or near-duplicate image. The procedure is not a fully transactional replacement if copying fails after validation. Audit duplicates and keep backups before a new experiment.

## 4. Preserve the trained input contract

**Decision:** use 128 × 128 RGB inputs, bilinear resizing, float32 values in 0–255, and the existing model's `Rescaling(1/255)` layer.

**Reason:** inspecting the saved architecture confirms that normalization is already embedded. Dividing pixels by 255 in the upload/CLI preprocessing as well would normalize twice and create a training/inference mismatch.

**Tradeoff:** square resizing stretches aspect ratios. Changing to cropping, padding, or another resolution should be treated as a separately evaluated training change. External photos receive EXIF orientation correction; Pillow and Keras decoding behavior are not claimed to be identical for every format.

## 5. Separate photo uploads from training formats

**Problem:** `test_images/class_images.jfif` was present but excluded by the CLI's JPEG/PNG extension filter, and Streamlit did not accept it.

**Decision:** share an inference extension list across CLI discovery and the uploader: JPEG, JFIF, PNG, WebP, BMP, and TIFF. Keep training formats limited to JPEG/PNG because the Keras directory loader does not index every Pillow-supported extension.

Uploaded streams are rewound before decoding. Grayscale and palette images are converted to RGB; EXIF orientation is applied; transparent photos are composited over white rather than exposing hidden RGB values as visible background. Corrupt files and excessive decoded dimensions produce actionable errors. TIFF inference uses the first frame. HEIC/HEIF is not supported.

**Tradeoff:** white is a practical background convention for product photos, not an accuracy claim. Its effect needs labeled validation data. The model weights are unchanged.

## 6. Load inference artifacts without training state

**Decision:** load the prediction model with `compile=False`; validate input shape, output count, and a nonempty list of unique label strings. Preserve the saved label order.

**Reason:** prediction does not need optimizer slots, training loss, or metrics. Restoring that state unnecessarily adds compatibility requirements. Validate output shape, finite values, probability bounds, and sum-to-one before rendering a result.

**Limitation:** output-count validation cannot detect a reordered list of the same labels. Artifact provenance is still necessary. Keep the model and label JSON together. Full training/evaluation continues to require compatible Keras dependencies.

## 7. Keep the baseline architecture and report its actual limits

The existing network uses Conv2D/MaxPooling blocks with 32, 64, and 128 filters, global average pooling, Dense(128), Dropout(0.3), and a six-class softmax. It has 110,534 trainable parameters. Global average pooling reduces dense-layer size compared with flattening the spatial tensor.

Training uses Adam at an initial learning rate of 0.001 and sparse categorical cross-entropy. Callbacks monitor validation loss: early stopping with patience 6, learning-rate reduction with patience 3, and the best model checkpoint. Augmentation includes horizontal flipping, rotation up to approximately 36 degrees, and 10% zoom.

The saved run completed 30 epochs, with its best validation loss at epoch 30. Recorded train accuracy was 72.29% and validation accuracy 69.54%, a 2.75 percentage-point gap. Training accuracy includes augmentation, so this difference alone is not a controlled measurement of overfitting. The recorded elapsed time is 2,185.9 seconds; hardware was not established.

Historical test accuracy is 67.95% on 702 images, with macro F1 0.6760. There are 225 recorded mistakes. Plastic is weakest by F1 (0.5657), while the largest confusion pair is metal → paper (26 images). These facts motivate better data and model experiments; they are not evidence that format fixes improved accuracy.

## 8. Failure analysis without fabricated observations

The recorded featured error is `dataset/test/organic/biological123.jpg`: organic predicted as paper at 98.17% confidence. Both classes had 542 training images according to the saved summary. This demonstrates that a high softmax score can accompany an incorrect label.

The original source image is absent from this checkout. No claim is made about its lighting, texture, composition, or the precise cause of the error. Similar material appearance or background reliance are hypotheses requiring inspection and controlled experiments.

The demonstration inputs also lack verified waste labels. The historical car example received glass at 98.40%; this is evidence of an out-of-scope confidence problem, not a successful car classification. A file's name is insufficient to establish benchmark ground truth.

## 9. Error reporting and verification

The app catches inference failures and displays an error rather than ending with an uncaught traceback. Batch CLI runs continue past invalid images, retain successful results, and return a nonzero status for partial or total failure. `--output` allows verification artifacts to be written without replacing historical predictions.

Filename-derived class hints require a separator or an exact match, avoiding false annotations such as treating `paperweight.jpg` as a confirmed paper example. These hints remain explicitly unverified.

Regression tests cover every bundled image, including JFIF; grayscale/palette/RGBA uploads; EXIF orientation; transparent pixels; damaged and oversized images; repeated deterministic inference; raw pixel scaling; invalid labels and probabilities; partial failures; and dataset preflight. Application tests exercise startup and prediction/error display. See [VERIFICATION.md](VERIFICATION.md) for executed checks and environment details.

## 10. Dependency compatibility

Runtime dependencies are pinned to the versions used for verification, including Keras 3.15.1, matching the saved artifact metadata. Pandas is declared directly because the app imports it; unused OpenCV was removed. The Windows verification environment required a shorter virtual-environment path after TensorFlow installation exceeded the platform path limit. This setup issue and its workaround are documented in the README.

## 11. Follow-up experiments

1. Recover dataset attribution, licenses, labels, and split manifests.
2. Build a representative labeled waste-photo validation set, including difficult backgrounds and mixed materials.
3. Compare this CNN with transfer learning using a documented, compatible preprocessing contract.
4. Evaluate calibration and unknown-input handling using both in-scope and out-of-scope data. A simple confidence threshold is insufficient.
5. Select improvements on validation data, then evaluate a held-out test set and update the recorded artifacts together.

No retraining, new accuracy claim, or general-object classifier is included in this repair.
