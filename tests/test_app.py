"""Run the real Streamlit script; inject uploads at the widget boundary."""

import io

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import predict as inference
from conftest import make_image_bytes
from utils import PROJECT_ROOT


class Upload(io.BytesIO):
    def __init__(self, contents):
        super().__init__(contents)
        self.size = len(contents)


def run_app():
    return AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90).run()


def test_app_starts_and_explains_scope():
    app = run_app()
    assert not app.exception
    assert not app.error
    assert any("only classifies waste" in warning.value for warning in app.warning)
    assert len(app.get("file_uploader")) == 1


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_app_displays_prediction(monkeypatch, fmt):
    uploaded = Upload(make_image_bytes(fmt))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = run_app()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 1
    assert app.metric[0].label == "Confidence"
    assert len(app.get("progress")) == 3


@pytest.mark.parametrize("oversized", [False, True])
def test_invalid_upload_shows_an_error(monkeypatch, oversized):
    uploaded = Upload(b"broken photo")
    if oversized:
        uploaded.size = 11 * 1024 * 1024
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = run_app()
    assert not app.exception
    assert len(app.error) == 1
    assert not app.metric


def test_inference_failure_is_displayed_without_crashing(monkeypatch):
    uploaded = Upload(make_image_bytes("JPEG"))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)

    def fail(*args, **kwargs):
        raise ValueError("invalid model output")

    monkeypatch.setattr(inference, "predict", fail)
    app = run_app()
    assert not app.exception
    assert "Could not classify" in app.error[0].value


def test_missing_model_shows_actionable_message(monkeypatch):
    st.cache_resource.clear()

    def fail():
        raise FileNotFoundError("Run: python src/train.py")

    monkeypatch.setattr(inference, "load_model_and_classes", fail)
    app = run_app()
    assert not app.exception
    assert "train.py" in app.error[0].value


def test_model_class_mismatch_is_shown_as_error(monkeypatch):
    st.cache_resource.clear()

    def fail():
        raise ValueError("Model output count (6) does not match class_names.json (12).")

    monkeypatch.setattr(inference, "load_model_and_classes", fail)
    app = run_app()
    assert not app.exception
    assert "output count" in app.error[0].value
    assert not app.metric
