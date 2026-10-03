"""Dataset preparation: invalid raw data must not erase a split, and 12 classes split cleanly."""

import numpy as np
import pytest
from PIL import Image

import prepare_dataset

CLASSES = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
           "metal", "paper", "plastic", "shoes", "trash", "white-glass"]


def patch_dirs(monkeypatch, tmp_path):
    for attr, name in [("RAW_DIR", "raw"), ("TRAIN_DIR", "train"),
                       ("VAL_DIR", "validation"), ("TEST_DIR", "test")]:
        monkeypatch.setattr(prepare_dataset, attr, tmp_path / name)


def test_preflight_preserves_previous_split(tmp_path, monkeypatch):
    patch_dirs(monkeypatch, tmp_path)
    (tmp_path / "raw" / "empty_class").mkdir(parents=True)
    (tmp_path / "train").mkdir()
    sentinel = tmp_path / "train" / "keep.txt"
    sentinel.write_text("existing split")
    with pytest.raises(ValueError, match="too few images"):
        prepare_dataset.main()
    assert sentinel.read_text() == "existing split"


def test_twelve_classes_are_split_70_15_15(tmp_path, monkeypatch):
    patch_dirs(monkeypatch, tmp_path)
    for name in CLASSES:
        folder = tmp_path / "raw" / name
        folder.mkdir(parents=True)
        for i in range(20):
            Image.fromarray(np.full((8, 8, 3), i, dtype="uint8")).save(folder / f"{i}.jpg")
    prepare_dataset.main()
    for name in CLASSES:
        assert len(list((tmp_path / "train" / name).iterdir())) == 14
        assert len(list((tmp_path / "validation" / name).iterdir())) == 3
        assert len(list((tmp_path / "test" / name).iterdir())) == 3
