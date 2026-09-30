"""Predict the class of external (unseen) images.

Usage:
    python src/predict.py test_images/plastic_bottle.jpg
    python src/predict.py                # every image inside test_images/
    python src/predict.py --save         # also write results/unseen_predictions.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

from utils import (MODEL_PATH, CLASS_NAMES_PATH, RESULTS_DIR, TEST_IMAGES_DIR,
                   PROJECT_ROOT, IMAGE_SIZE, VALID_EXTENSIONS)

LOW_CONFIDENCE = 0.60


def load_model_and_classes():
    """Load the trained model and its class names."""
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Trained model or class_names.json not found in models/.\n"
            "Run: python src/train.py")
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def load_image(source):
    """Open an image (path or uploaded file) and return it as an RGB PIL image."""
    try:
        with Image.open(source) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read the image: {error}") from error


def preprocess(img):
    """Resize like the training pipeline. Normalization happens inside the model."""
    array = np.asarray(img, dtype="float32")             # values stay in 0-255
    array = tf.image.resize(array, IMAGE_SIZE).numpy()   # bilinear, same as training loader
    return np.expand_dims(array, axis=0)


def predict(model, class_names, img):
    """Return [(class_name, probability), ...] sorted from most to least likely."""
    probabilities = model.predict(preprocess(img), verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return [(class_names[i], float(probabilities[i])) for i in order]


def expected_class(path, class_names):
    """Guess the true class from the file name (e.g. plastic_bottle.jpg)."""
    stem = path.stem.lower()
    for name in sorted(class_names, key=len, reverse=True):
        if stem.startswith(name.lower()):
            return name
    return None


def relative_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_paths(arguments):
    if arguments:
        return [Path(a) for a in arguments]
    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {TEST_IMAGES_DIR}")
    paths = sorted(p for p in TEST_IMAGES_DIR.iterdir()
                   if p.suffix.lower() in VALID_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(
            f"No .jpg/.jpeg/.png images found in {TEST_IMAGES_DIR}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Classify unseen images.")
    parser.add_argument("images", nargs="*", help="image path(s); default: test_images/")
    parser.add_argument("--save", action="store_true",
                        help="save results to results/unseen_predictions.json")
    args = parser.parse_args()

    try:
        paths = collect_paths(args.images)
        model, class_names = load_model_and_classes()
    except FileNotFoundError as error:
        sys.exit(f"Error: {error}")

    results = []
    for path in paths:
        print("\n" + "-" * 40)
        print(f"Image: {path.name}")
        if not path.exists():
            print("  Error: file not found.")
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            print(f"  Error: unsupported format. Use {sorted(VALID_EXTENSIONS)}")
            continue
        try:
            ranked = predict(model, class_names, load_image(path))
        except ValueError as error:
            print(f"  Error: {error}")
            continue

        label, confidence = ranked[0]
        expected = expected_class(path, class_names)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")
        if expected:
            verdict = "CORRECT" if expected == label else "WRONG"
            print(f"Expected (from file name): {expected} -> {verdict}")
        if confidence < LOW_CONFIDENCE:
            print("Note: low confidence, treat this prediction with caution.")
        print("\nTop Predictions:")
        for name, probability in ranked:
            print(f"  {name:<12} {probability * 100:6.2f}%")

        results.append({
            "file": relative_path(path),
            "expected": expected,
            "predicted": label,
            "confidence": round(confidence, 4),
            "correct": None if expected is None else expected == label,
            "all_probabilities": {n: round(p, 4) for n, p in ranked},
        })

    if not results:
        sys.exit("\nNo image could be classified.")

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "unseen_predictions.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\nSaved: results/unseen_predictions.json")


if __name__ == "__main__":
    main()