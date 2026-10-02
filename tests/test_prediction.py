"""Regression tests exercise real decoding and the bundled model on CPU."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import predict as inference
from utils import INFERENCE_EXTENSIONS, TEST_IMAGES_DIR


@pytest.fixture(scope="session")
def classifier():
    return inference.load_model_and_classes()


@pytest.mark.parametrize("filename", sorted(p.name for p in TEST_IMAGES_DIR.iterdir()))
def test_every_bundled_photo_is_discovered_and_classified(classifier, filename):
    path = TEST_IMAGES_DIR / filename
    assert path in inference.collect_paths([])
    model, classes = classifier
    ranked = inference.predict(model, classes, inference.load_image(path))
    assert len(ranked) == len(classes)
    assert {name for name, _ in ranked} == set(classes)
    values = [value for _, value in ranked]
    assert values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
def test_uploaded_stream_is_reusable_and_converted_to_rgb(mode):
    stream = io.BytesIO()
    Image.new(mode, (7, 11)).save(stream, format="PNG")
    stream.seek(0, 2)  # An upload can already have been read by the UI.
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


def test_preprocessing_keeps_raw_pixel_scale_for_model_normalization():
    batch = inference.preprocess(Image.new("L", (8, 12), 255))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(batch, 255)


def test_repeated_inference_has_no_training_augmentation(classifier):
    model, classes = classifier
    image = inference.load_image(TEST_IMAGES_DIR / "class_car.jpg")
    first = inference.predict(model, classes, image)
    second = inference.predict(model, classes, image)
    assert first == second


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


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        inference.load_model_and_classes(tmp_path / "missing.keras")


def test_filename_label_requires_a_boundary():
    assert inference.expected_class(Path("paper_cup.jpg"), ["paper"]) == "paper"
    assert inference.expected_class(Path("paper.jpg"), ["paper"]) == "paper"
    assert inference.expected_class(Path("paperweight.jpg"), ["paper"]) is None


def test_batch_reports_partial_failure_and_saves_successes(tmp_path, monkeypatch, classifier):
    output = tmp_path / "predictions.json"
    monkeypatch.setattr(inference, "load_model_and_classes", lambda: classifier)
    monkeypatch.setattr(sys, "argv", ["predict.py", str(TEST_IMAGES_DIR / "class_images.jfif"),
                                     str(tmp_path / "missing.jpg"), "--output", str(output)])
    with pytest.raises(SystemExit, match="1 image"):
        inference.main()
    results = json.loads(output.read_text())
    assert len(results) == 1
    assert results[0]["file"].endswith("class_images.jfif")


def test_discovery_ignores_directories_with_image_extensions(tmp_path, monkeypatch):
    (tmp_path / "folder.jpg").mkdir()
    monkeypatch.setattr(inference, "TEST_IMAGES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No supported images"):
        inference.collect_paths([])
