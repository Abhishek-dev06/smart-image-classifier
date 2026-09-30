# Smart Image Classifier

A convolutional neural network that classifies images of <dataset topic, e.g. waste>
into <N> categories, with a Streamlit upload interface. Built for the GDG-USAR
tech team task.

## Overview

This project covers the full image-classification pipeline: dataset analysis,
train/validation/test split, preprocessing, augmentation, baseline CNN training,
evaluation (accuracy, precision, recall, F1, confusion matrix), testing on unseen
images, error analysis, and a simple web interface.

## Features

- Image classification with a small CNN built in TensorFlow/Keras
- Resize, RGB conversion and 0-1 normalization
- Training-time augmentation (flip, rotation, zoom)
- Prediction with softmax confidence and top-3 predictions
- Confusion matrix, classification report and error analysis
- Streamlit app for uploading an image and getting a prediction

## Project Architecture

```text
smart-image-classifier/
├── dataset/              # not included in the repo (see Dataset)
│   ├── raw/
│   ├── train/
│   ├── validation/
│   └── test/
├── test_images/          # unseen images for demonstration
├── models/               # image_classifier.keras, class_names.json
├── results/              # plots and metrics
├── src/
│   ├── utils.py
│   ├── prepare_dataset.py
│   ├── dataset_analysis.py
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── confusion_analysis.py
│   ├── predict.py
│   ├── error_analysis.py
│   └── generate_results_section.py
├── app.py
├── requirements.txt
├── README.md
├── DECISIONS.md
└── LICENSE
```

## Dataset

<Dataset name> with <N> classes: <class names>. Source: <link>.
The dataset is not redistributed in this repository. Download it from the link
above and place the class folders inside `dataset/raw/`, so that you have for
example `dataset/raw/<class_1>/`, `dataset/raw/<class_2>/`, and so on.

Then create the splits (70% train, 15% validation, 15% test, fixed seed 42):

```bash
python src/prepare_dataset.py
```

## Installation

```bash
git clone <repository-url>
cd smart-image-classifier
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Tested with Python <your version>. Python 3.10-3.12 is recommended for TensorFlow.

## Usage

```bash
python src/dataset_analysis.py      # dataset summary and class distribution
python src/train.py                 # train the model
python src/evaluate.py              # test-set metrics
python src/confusion_analysis.py    # confusion matrix
python src/predict.py test_images/<image>.jpg
python src/error_analysis.py        # misclassified images
streamlit run app.py                # web interface
```

A trained model is included in `models/`, so `predict.py` and the Streamlit app
work without retraining. <Remove this line if you did not commit the model.>

## Web Application

![App screenshot](results/app_screenshot.png)

## Model

Baseline CNN: three Conv2D + MaxPooling blocks (32, 64, 128 filters),
GlobalAveragePooling, Dense(128), Dropout(0.3) and a softmax output layer.
Optimizer: Adam. Loss: sparse categorical cross-entropy. Input size: 128 x 128.
Augmentation and normalization are layers inside the model, so augmentation is
active only during training.

<PASTE results_section.md HERE>

## Data Leakage Prevention

- The data is split into train/validation/test once, before training.
- Only the training data is augmented. Augmentation layers are inactive during
  validation, testing and prediction.
- Callbacks (early stopping, checkpointing) monitor validation loss only.
- The test set is used only for the final evaluation.
- Images in `test_images/` are separate from the dataset.

## Reproducibility

Random seeds are fixed (seed 42) for Python, NumPy and TensorFlow. Small numerical
differences can still occur across CPU/GPU types and library versions.

## Limitations

- <Write honestly, based on your results. Examples below, keep only what is true.>
- The model knows only its <N> classes. Any other object will still be assigned to one of them.
- Softmax confidence is not a calibrated probability, so a wrong prediction can still show high confidence.
- <Class imbalance: mention your imbalance ratio and its effect if it applies.>
- <The dataset is small and images may not cover all lighting, backgrounds and angles.>
- <Duplicate or near-duplicate images in the public dataset could make test results look better than they are.>
- Images are resized to a square, which can distort the aspect ratio.

## Future Improvements

- Transfer learning (for example MobileNetV2 with ImageNet weights)
- More and more varied training data
- Class balancing
- Hyperparameter tuning
- Better confidence calibration

## Author

<Your name>, <college>. GDG-USAR tech team task.

## License

MIT License. See `LICENSE`.