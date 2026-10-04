"""Regression tests: real decoding, real model on CPU, generated images."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import predict as inference
from conftest import make_image_bytes
from utils import CLASS_NAMES_PATH, INFERENCE_EXTENSIONS, MODEL_METADATA_PATH

EXPECTED_CLASSES = ["battery", "glass", "metal", "organic", "paper", "plastic"]


# ---- bundled artifacts must agree with one another ----

def test_bundled_artifacts_match(classifier):
    model, classes = classifier
    assert classes == EXPECTED_CLASSES
    assert model.output_shape == (None, 6)
    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    assert json.loads(CLASS_NAMES_PATH.read_text()) == metadata["class_names"]
    assert metadata["number_of_classes"] == 6


# ---- inference on every supported format ----

@pytest.mark.parametrize("filename", ["a.jpg", "b.jfif", "c.png", "d.webp", "e.bmp", "f.tiff"])
def test_every_supported_photo_is_discovered_and_classified(classifier, photo_dir, monkeypatch,
                                                            filename):
    monkeypatch.setattr(inference, "TEST_IMAGES_DIR", photo_dir)
    path = photo_dir / filename
    assert path in inference.collect_paths([])
    model, classes = classifier
    ranked = inference.predict(model, classes, inference.load_image(path))
    assert len(ranked) == len(classes) == 6
    assert {name for name, _ in ranked} == set(classes)
    values = [value for _, value in ranked]
    assert values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
def test_uploaded_stream_is_reusable_and_converted_to_rgb(mode):
    stream = io.BytesIO()
    Image.new(mode, (7, 11)).save(stream, format="PNG")
    stream.seek(0, 2)
    image = inference.load_image(stream)
    assert image.mode == "RGB"
    assert image.size == (7, 11)
    assert np.array_equal(image, inference.load_image(stream))


@pytest.mark.parametrize("suffix,fmt", [(".jfif", "JPEG"), (".WEBP", "WEBP"),
                                       (".bmp", "BMP"), (".tiff", "TIFF")])
def test_additional_photo_formats(tmp_path, suffix, fmt):
    path = tmp_path / ("photo" + suffix)
    Image.new("RGB", (10, 20), "red").save(path, format=fmt)
    assert path.suffix.lower() in INFERENCE_EXTENSIONS
    assert inference.load_image(path).size == (10, 20)


def test_transparency_is_composited_on_white():
    stream = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(stream, format="PNG")
    assert inference.load_image(stream).getpixel((0, 0)) == (255, 255, 255)


def test_camera_orientation_is_applied():
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (7, 11)).save(stream, format="JPEG", exif=exif)
    assert inference.load_image(stream).size == (11, 7)


def test_corrupt_upload_is_reported():
    with pytest.raises(ValueError, match="Could not read"):
        inference.load_image(io.BytesIO(b"not an image"))


def test_oversized_decoded_image_is_reported(monkeypatch):
    stream = io.BytesIO()
    Image.new("RGB", (10, 10)).save(stream, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 60)
    with pytest.raises(ValueError, match="Could not read"):
        inference.load_image(stream)


def test_preprocessing_keeps_raw_pixel_scale_for_model_normalization(classifier):
    model, _ = classifier
    target_size = inference.get_model_input_size(model)
    batch = inference.preprocess(Image.new("L", (8, 12), 255), target_size=target_size)
    assert batch.shape == (1, *target_size, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch, 255)


def test_repeated_inference_has_no_training_augmentation(classifier):
    model, classes = classifier
    image = inference.load_image(io.BytesIO(make_image_bytes("PNG")))
    first = inference.predict(model, classes, image)
    second = inference.predict(model, classes, image)
    assert first == second


# ---- validation ----

@pytest.mark.parametrize("output", [[[1.0]], [[float("nan"), 0]],
                                    [[0.8, 0.8]], [[-0.1, 1.1]]])
def test_invalid_model_outputs_are_rejected(output):
    model = SimpleNamespace(predict=lambda *args, **kwargs: output)
    with pytest.raises(ValueError, match="Model"):
        inference.predict(model, ["glass", "paper"], Image.new("RGB", (2, 2)))


@pytest.mark.parametrize("labels", [[], {}, ["glass", "glass"], [""], [1]])
def test_invalid_labels_are_rejected_before_model_load(tmp_path, labels):
    model = tmp_path / "model.keras"
    model.touch()
    names = tmp_path / "classes.json"
    names.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="unique class names"):
        inference.load_model_and_classes(model, names)


def test_model_and_class_count_must_match(tmp_path):
    names = tmp_path / "classes.json"
    names.write_text('["glass"]')
    with pytest.raises(ValueError, match="output count"):
        inference.load_model_and_classes(class_names_path=names)


def test_class_order_must_match_metadata(tmp_path):
    names = tmp_path / "classes.json"
    names.write_text(json.dumps(list(reversed(EXPECTED_CLASSES))))
    with pytest.raises(ValueError, match="different classes or a different order"):
        inference.load_model_and_classes(class_names_path=names)


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        inference.load_model_and_classes(tmp_path / "missing.keras")


def test_missing_class_names_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="class_names.json"):
        inference.load_model_and_classes(class_names_path=tmp_path / "missing.json")


# ---- CLI helpers ----

def test_filename_label_requires_a_boundary():
    assert inference.expected_class(Path("paper_cup.jpg"), ["paper"]) == "paper"
    assert inference.expected_class(Path("paper.jpg"), ["paper"]) == "paper"
    assert inference.expected_class(Path("paperweight.jpg"), ["paper"]) is None


def test_batch_reports_partial_failure_and_saves_successes(tmp_path, monkeypatch, classifier,
                                                           photo_dir):
    output = tmp_path / "predictions.json"
    monkeypatch.setattr(inference, "load_model_and_classes", lambda: classifier)
    monkeypatch.setattr(sys, "argv", ["predict.py", str(photo_dir / "b.jfif"),
                                     str(tmp_path / "missing.jpg"), "--output", str(output)])
    with pytest.raises(SystemExit, match="1 image"):
        inference.main()
    results = json.loads(output.read_text())
    assert len(results) == 1
    assert results[0]["file"].endswith("b.jfif")
    assert len(results[0]["all_probabilities"]) == 6


def test_discovery_ignores_directories_with_image_extensions(tmp_path, monkeypatch):
    (tmp_path / "folder.jpg").mkdir()
    monkeypatch.setattr(inference, "TEST_IMAGES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No supported images"):
        inference.collect_paths([])
