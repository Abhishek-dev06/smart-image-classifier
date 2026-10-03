# Smart Image Classifier (WasteWealth MobileNetV2)

A TensorFlow/Keras app that classifies **waste photos** into **12 material categories** using a MobileNetV2 transfer-learning model. It has a Streamlit upload interface, a REST API and a command-line tool.

**Categories (12):** `battery` · `biological` · `brown-glass` · `cardboard` · `clothes` · `green-glass` · `metal` · `paper` · `plastic` · `shoes` · `trash` · `white-glass`

> This is a waste classifier, not a general object recognizer. The model always picks one of its 12 classes, so unrelated photos (cars, people, screenshots) still get a waste label, sometimes with high confidence.

## Quick start

Python 3.11 is recommended.

```sh
git clone https://github.com/Abhishek-dev06/smart-image-classifier.git
cd smart-image-classifier
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a JPEG, JFIF, PNG, WebP, BMP or TIFF photo (up to 10 MB). Transparent images are placed on a white background.

## Model files (important)

The app needs these three files in `models/`, **all from the same training run**:

| File | Purpose |
| --- | --- |
| `image_classifier.keras` | Trained model with 12 softmax outputs |
| `class_names.json` | Ordered labels (alphabetical, as produced by Keras) |
| `model_metadata.json` | Name, input size and the same label order |

If the model output count differs from `class_names.json`, the app shows `Model output count (N) does not match class_names.json (12)`. That means `image_classifier.keras` comes from a different run (for example the old 6-class model). Replace it with the 12-class model, commit it, and reboot the Streamlit app so the cached model is reloaded.

`train.py` writes all three files together. Do not edit or reorder labels by hand.

## REST API

```sh
python api.py
```

| Endpoint | Description |
| --- | --- |
| `GET /` | Interactive tester |
| `GET /api/health` | Status |
| `GET /api/classes` | The 12 classes |
| `GET /api/model` | Model metadata |
| `POST /api/predict` | Multipart `file`/`image` or JSON `image_base64` (max 10 MB) |

```sh
curl -X POST -F "file=@path/to/waste-photo.jpg" http://localhost:5000/api/predict
```

## Command line

```sh
python src/predict.py path/to/photo.jpg           # one image
python src/predict.py                             # every image in test_images/
python src/predict.py --save                      # also write results/unseen_predictions.json
python src/predict.py photo.jpg --output out.json # custom output file
```

Output lists all 12 probabilities. A top score below `0.60` prints a caution; this is a display heuristic, not a calibrated threshold. A file name like `plastic_bottle.jpg` gives an unverified expected-class hint.

## Training (12 classes)

Arrange labeled JPEG/PNG images (at least 10 per class):

```text
dataset/raw/<class name>/*.jpg     # 12 folders, names as listed above
```

```sh
python src/prepare_dataset.py      # 70/15/15 split, seed 42, replaces old splits
python src/dataset_analysis.py
python src/data_pipeline.py
python src/train.py --epochs 30    # --backbone baseline_cnn for the small CNN
python src/evaluate.py
python src/confusion_analysis.py
python src/error_analysis.py
python src/generate_results_section.py
```

Images are resized to 224 x 224 RGB. Pixel values stay in 0-255; the model's built-in `Rescaling` layer normalizes them once, so do not divide by 255 again outside the model. Augmentation (flip, rotation, zoom) runs only during training.

Training and evaluation overwrite files in `models/` and `results/`. Back them up first.

## Results

Metrics are produced by `evaluate.py` into `results/evaluation_metrics.json` and `results/training_summary.json`. Run `python src/generate_results_section.py` to build `results/results_section.md`, then paste the generated tables here. Only use numbers from the 12-class run.

## Project layout

```text
app.py                 Streamlit interface
api.py                 REST API
models/                Model, class_names.json, model_metadata.json
results/               Metrics, plots, predictions
src/                   Data, training, evaluation and inference scripts
test_images/           Your own waste photos for quick checks (not a benchmark)
tests/                 pytest regression tests
requirements.txt       Runtime dependencies
requirements-dev.txt   Runtime + pytest
DECISIONS.md           Design decisions
```

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests generate their own images, so they do not need `test_images/` or the training dataset. They need the 12-class model in `models/`: one test fails on purpose if `image_classifier.keras` does not have 12 outputs or does not match `class_names.json`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Model output count (6) does not match class_names.json (12)` | Wrong `.keras` file. Use the 12-class model from your latest training run. |
| Old model still shown on Streamlit Cloud | Manage app, then Reboot app (the model is cached). |
| `.keras` not on GitHub | Check `.gitignore`; files over 100 MB need Git LFS. |
| Model fails to load | Keep the pinned `tensorflow` and `keras` versions the model was trained with. |
| Wrong labels but no error | `class_names.json` order differs from training order. |
| HEIC/HEIF upload fails | Export as JPEG or PNG first. |

## License

Code: [MIT License](LICENSE). Dataset and sample-image rights need separate checks.
