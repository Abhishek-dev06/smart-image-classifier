# Smart Image Classifier

A TensorFlow/Keras waste-image classifier with a Streamlit upload interface, REST API, CLI inference, training scripts and regression tests.

## Current bundled model

The model currently committed at `models/image_classifier.keras` is the original **6-class** waste classifier. Its matching classes are:

`battery` · `glass` · `metal` · `organic` · `paper` · `plastic`

The repository also contains code for a newer **12-class WasteWealth MobileNetV2** training configuration. Do not mix the 12-class label/metadata files with the bundled 6-output model. A model file, `class_names.json`, and `model_metadata.json` must always come from the same training run.

> This is a waste classifier, not a general object recognizer. Unrelated photos such as cars, people or screenshots will still be forced into one of the known waste classes.

## Quick start

Python 3.11 is recommended.

```sh
git clone https://github.com/Abhishek-dev06/smart-image-classifier.git
cd smart-image-classifier
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Linux/macOS:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a JPEG, JFIF, PNG, WebP, BMP or TIFF image up to 10 MB.

## Model artifact rule

These three files must match each other:

| File | Purpose |
| --- | --- |
| `models/image_classifier.keras` | Trained Keras model |
| `models/class_names.json` | Labels in exact model-output order |
| `models/model_metadata.json` | Model name, input shape and the same labels |

The application validates the output count against `class_names.json`. If they differ, inference stops instead of silently attaching the wrong label to a prediction.

### Current artifact set

- Model outputs: 6
- Classes: `battery`, `glass`, `metal`, `organic`, `paper`, `plastic`
- Metadata: aligned to the bundled 6-class model

### Upgrading to the 12-class model

The training code supports these 12 classes:

`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`

To actually switch the application to 12 classes, train a new model on the 12-class dataset and replace **all three artifacts together**. Merely changing JSON labels does not convert a 6-output neural network into a 12-output network.

## REST API

```sh
python api.py
```

| Endpoint | Description |
| --- | --- |
| `GET /` | Interactive tester |
| `GET /api/health` | API/model status |
| `GET /api/classes` | Active model classes |
| `GET /api/model` | Model metadata |
| `POST /api/predict` | Predict from multipart image or base64 JSON |

Example:

```sh
curl -X POST -F "file=@path/to/waste-photo.jpg" http://localhost:5000/api/predict
```

## Command line

```sh
python src/predict.py path/to/photo.jpg
python src/predict.py
python src/predict.py --save
python src/predict.py photo.jpg --output out.json
```

The CLI prints probabilities for every active class. A top probability below `0.60` is displayed as low confidence; this is a UI heuristic, not a calibrated guarantee.

## Training a new 12-class model

Arrange labeled images like this:

```text
dataset/raw/<class-name>/*.jpg
```

Then run:

```sh
python src/prepare_dataset.py
python src/dataset_analysis.py
python src/data_pipeline.py
python src/train.py --epochs 30
python src/evaluate.py
python src/confusion_analysis.py
python src/error_analysis.py
python src/generate_results_section.py
```

`src/train.py` is intended to write the model and its matching class metadata together. Commit the newly generated artifact set only after confirming that the output count equals the class count.

## Project layout

```text
app.py                 Streamlit interface
api.py                 REST API
models/                Model + matching labels/metadata
results/               Metrics, plots and predictions
src/                   Data, training, evaluation and inference code
test_images/           Optional local test photos
tests/                 pytest regression tests
requirements.txt       Runtime dependencies
requirements-dev.txt   Runtime + pytest dependencies
```

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The regression tests check that the bundled model output count, class list and metadata agree before testing inference.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Model output count (...) does not match class_names.json (...)` | Use model, labels and metadata from the same training run. |
| Old model still shown by Streamlit | Clear/reboot the deployed app so the cached model reloads. |
| `.keras` missing | Restore/train the model; large models may require Git LFS. |
| Wrong labels but no load error | Verify label order is exactly the training output order. |
| HEIC/HEIF upload fails | Export the image as JPEG or PNG. |

## License

Code: [MIT License](LICENSE). Dataset and sample-image rights should be checked separately.
