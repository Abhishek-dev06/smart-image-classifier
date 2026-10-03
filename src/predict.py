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
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

from utils import (MODEL_PATH, CLASS_NAMES_PATH, MODEL_METADATA_PATH, RESULTS_DIR, TEST_IMAGES_DIR,
                   PROJECT_ROOT, IMAGE_SIZE, INFERENCE_EXTENSIONS)

LOW_CONFIDENCE = 0.60


def load_metadata(metadata_path=MODEL_METADATA_PATH):
    """Load model metadata if present."""
    metadata_path = Path(metadata_path)
    if metadata_path.is_file():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_model_input_size(model):
    """Extract (height, width) from model input shape if defined, otherwise IMAGE_SIZE."""
    if hasattr(model, "input_shape") and model.input_shape and len(model.input_shape) == 4:
        h, w = model.input_shape[1], model.input_shape[2]
        if h and w:
            return (int(h), int(w))
    return IMAGE_SIZE


def load_model_and_classes(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH,
                           metadata_path=MODEL_METADATA_PATH):
    """Load the trained model and its class names."""
    model_path, class_names_path = Path(model_path), Path(class_names_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Copy your trained image_classifier.keras into models/ or run: python src/train.py")
    if not class_names_path.is_file():
        raise FileNotFoundError(
            f"class_names.json not found: {class_names_path}\n"
            "Copy the class_names.json from the same training run into models/.")
    with open(class_names_path, encoding="utf-8") as f:
        class_names = json.load(f)
    if (not isinstance(class_names, list) or not class_names
            or any(not isinstance(name, str) or not name.strip() for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError("class_names.json must contain a nonempty list of unique class names.")
    metadata = load_metadata(metadata_path)
    # Inference does not need saved optimizer state or training metrics.
    model = keras.models.load_model(model_path, compile=False)
    input_size = get_model_input_size(model)
    if tuple(model.input_shape) != (None, *input_size, 3):
        raise ValueError(f"Model must accept RGB images of size {input_size}.")
    if tuple(model.output_shape) != (None, len(class_names)):
        model_name = metadata.get("model_name", "WasteWealth MobileNetV2") if metadata else "WasteWealth MobileNetV2"
        out_count = model.output_shape[-1] if model.output_shape else "unknown"
        raise ValueError(
            f"Model output count ({out_count}) does not match class_names.json ({len(class_names)}). "
            f"The file {model_path.name} is from a different training run. "
            f"Replace it with the {model_name} model trained on the same {len(class_names)} classes."
        )
    # class_names.json and model_metadata.json must describe the same labels in the same order.
    meta_names = metadata.get("class_names") if metadata else None
    if meta_names is not None and list(meta_names) != class_names:
        raise ValueError(
            "class_names.json and model_metadata.json list different classes or a different order. "
            "Use both files from the same training run.")
    return model, class_names


def load_image(source):
    """Open an image (path or uploaded file) and return it as an RGB PIL image."""
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as img:
                image = ImageOps.exif_transpose(img)
                # Match the usual white background of transparent product photos.
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba)
                return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError(f"Could not read the image: {error}") from error


def preprocess(img, target_size=None):
    """Resize like the training pipeline. Normalization happens inside the model."""
    if target_size is None:
        target_size = IMAGE_SIZE
    array = np.asarray(img.convert("RGB"), dtype="float32")  # values stay in 0-255
    array = tf.image.resize(array, target_size).numpy()   # bilinear, same as training loader
    return np.expand_dims(array, axis=0)


def predict(model, class_names, img):
    """Return [(class_name, probability), ...] sorted from most to least likely."""
    target_size = get_model_input_size(model)
    output = np.asarray(model.predict(preprocess(img, target_size=target_size), verbose=0))
    if output.shape != (1, len(class_names)) or not class_names:
        raise ValueError("Model prediction shape does not match the class names.")
    probabilities = output[0]
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-4)):
        raise ValueError("Model returned invalid probabilities; check the model artifact.")
    order = np.argsort(probabilities)[::-1]
    return [(class_names[i], float(probabilities[i])) for i in order]


def expected_class(path, class_names):
    """Guess the true class from the file name (e.g. plastic_bottle.jpg)."""
    stem = path.stem.lower()
    for name in sorted(class_names, key=len, reverse=True):
        label = name.lower()
        if stem == label or any(stem.startswith(label + sep) for sep in ("_", "-", " ")):
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
                   if p.is_file() and p.suffix.lower() in INFERENCE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(
            f"No supported images found in {TEST_IMAGES_DIR}: {sorted(INFERENCE_EXTENSIONS)}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Classify unseen images.")
    parser.add_argument("images", nargs="*", help="image path(s); default: test_images/")
    parser.add_argument("--save", action="store_true",
                        help="save results to results/unseen_predictions.json")
    parser.add_argument("--output", type=Path,
                        help="save predictions to a custom JSON path (implies --save)")
    args = parser.parse_args()

    try:
        paths = collect_paths(args.images)
        model, class_names = load_model_and_classes()
    except (OSError, ValueError, TypeError) as error:
        sys.exit(f"Error: {error}")

    metadata = load_metadata()
    model_name = metadata.get("model_name", "WasteWealth MobileNetV2") if metadata else "WasteWealth MobileNetV2"
    print(f"{model_name}. Supported classes ({len(class_names)}): " + ", ".join(class_names))
    print("Other subjects still receive a waste label, even at high confidence.")
    results = []
    failed = 0
    for path in paths:
        print("\n" + "-" * 40)
        print(f"Image: {path.name}")
        if not path.is_file():
            print("  Error: file not found.")
            failed += 1
            continue
        if path.suffix.lower() not in INFERENCE_EXTENSIONS:
            print(f"  Error: unsupported format. Use {sorted(INFERENCE_EXTENSIONS)}")
            failed += 1
            continue
        try:
            ranked = predict(model, class_names, load_image(path))
        except (OSError, ValueError, tf.errors.OpError) as error:
            print(f"  Error: {error}")
            failed += 1
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

    if args.save or args.output:
        output_path = args.output or RESULTS_DIR / "unseen_predictions.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_path}")
    if failed:
        sys.exit(f"\n{failed} image(s) could not be classified.")


if __name__ == "__main__":
    main()
