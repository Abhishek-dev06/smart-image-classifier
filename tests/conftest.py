"""Shared helpers: tests generate their own images, so they do not depend on test_images/."""

import io

import numpy as np
import pytest
from PIL import Image

import predict as inference


def make_image_bytes(fmt="JPEG", size=(48, 64), seed=0):
    """Return the bytes of a small random RGB photo in the given Pillow format."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype="uint8")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def classifier():
    """The real model from models/ (must be the 12-class model)."""
    return inference.load_model_and_classes()


@pytest.fixture
def photo_dir(tmp_path):
    """A folder with one photo per supported inference format."""
    for name, fmt in [("a.jpg", "JPEG"), ("b.jfif", "JPEG"), ("c.png", "PNG"),
                      ("d.webp", "WEBP"), ("e.bmp", "BMP"), ("f.tiff", "TIFF")]:
        (tmp_path / name).write_bytes(make_image_bytes(fmt))
    return tmp_path
