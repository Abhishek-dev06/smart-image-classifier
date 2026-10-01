# Verification Report

Verified on 2026-10-01 against the original repository commit `d17b9b080043a941b83973535517af687f645459`.

## Environment

Windows, Python 3.11.6, CPU inference. Runtime versions: TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, Pillow 12.3.0, pandas 3.0.6, Matplotlib 3.11.2, scikit-learn 1.9.1, Streamlit 1.64.0. Test runner: pytest 9.1.1.

The test subprocesses used `TF_NUM_INTRAOP_THREADS=2` and `TF_NUM_INTEROP_THREADS=2` to limit CPU thread use. TensorFlow was installed in a shorter workspace path after its nested header files exceeded the Windows path limit in the original temporary environment. No operating-system settings were changed.

## Reproduced defect

Running the untouched original CLI against `test_images/class_images.jfif` returned exit code 1 with:

```text
Error: unsupported format. Use ['.jpeg', '.jpg', '.png']
No image could be classified.
```

The fixed CLI discovered and processed the JFIF photo alongside all five other bundled images. The Streamlit test also exercised a JFIF upload and verified that the confidence metric and three prediction progress bars were rendered.

## Executed checks

| Check | Result |
| --- | --- |
| `python -m pytest -q` | **41 passed**, no failures or errors |
| Real bundled model on every demonstration image | **6/6 inputs processed**; each output contained six finite, normalized probabilities |
| Repeated inference | Identical results across repeated calls in the tested environment |
| Input handling | Grayscale, palette, RGBA, EXIF rotation, transparent background, JFIF, WebP, BMP, TIFF passed |
| Failure handling | Corrupt/oversized uploads, missing model, malformed labels, class-count mismatch, invalid probabilities, and partial batch failure passed |
| Streamlit AppTest | Startup, real-model JFIF prediction, invalid uploads, inference error, and missing-model display passed |
| Dataset preflight | Existing split preserved when raw data validation fails |
| Saved historical accuracy consistency | 477 correct / 702 = 67.9487%, agreeing with recorded 0.6795 |
| Dependency consistency | `python -m pip check`: no broken requirements found |
| Documentation links | All local links in README, decision record, and verification report resolve |
| Python compilation | Application, source scripts, and tests compiled successfully |

AppTest executes the Streamlit script with upload bytes injected at the widget boundary. This is application-level verification, not a manual browser test or a deployed-server test.

## Fresh inference output

The following predictions are recorded in [verification_predictions.json](results/verification_predictions.json). These images have no verified waste labels, so **processing success does not establish prediction correctness**.

| Image | Predicted waste label | Softmax score |
| --- | --- | ---: |
| class_car.jpg | glass | 98.40% |
| class_favicon.jpeg | paper | 48.15% |
| class_images.jfif | organic | 87.66% |
| class_pihu.png | organic | 63.70% |
| class_plush.png | organic | 97.72% |
| class_sshot.png | battery | 78.98% |

The car example still receives a high-confidence waste label, illustrating the model's closed-set limitation. The repair supports valid image formats and makes scope/errors explicit; it does not add general-object recognition or unknown-class detection.

## Artifact integrity and validation limits

The model file is byte-for-byte unchanged from the original checkout. SHA-256:

```text
146b1a45d4f5c4f6a50ed49746591fe5ee78ededa85d5614ba80626e4ba3a36a
```

The original class labels, training metrics, test metrics, and historical unseen predictions are preserved. The generated results section now labels those figures as historical and no longer asserts undocumented hardware or test-set use history.

The raw dataset, exact dataset source, and split manifest are absent. Training and full test-set evaluation were not rerun, and no new accuracy claim is made. Windows CPU behavior was tested; Linux/macOS, GPU execution, deployment, and real-world accuracy on a newly labeled waste-photo set remain unverified.

## Reproduce locally

From the repository root in a Python 3.11 environment:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python src/predict.py --output results/local_smoke_predictions.json
python -m pip check
python -m compileall -q app.py src tests
```
