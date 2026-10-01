"""Invalid raw data must not erase an existing training split."""

import pytest

import prepare_dataset


def test_preflight_preserves_previous_split(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "empty_class").mkdir(parents=True)
    train = tmp_path / "train"
    train.mkdir()
    sentinel = train / "keep.txt"
    sentinel.write_text("existing split")
    monkeypatch.setattr(prepare_dataset, "RAW_DIR", raw)
    monkeypatch.setattr(prepare_dataset, "TRAIN_DIR", train)
    monkeypatch.setattr(prepare_dataset, "VAL_DIR", tmp_path / "validation")
    monkeypatch.setattr(prepare_dataset, "TEST_DIR", tmp_path / "test")
    with pytest.raises(ValueError, match="too few images"):
        prepare_dataset.main()
    assert sentinel.read_text() == "existing split"
