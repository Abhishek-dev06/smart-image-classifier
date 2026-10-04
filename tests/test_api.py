"""REST API: health, classes, prediction, and error responses."""

import base64
import io

import pytest

import api
from conftest import make_image_bytes


@pytest.fixture
def client():
    return api.app.test_client()


def test_health_and_classes_match_bundled_model(client):
    assert client.get("/api/health").get_json()["number_of_classes"] == 6
    assert client.get("/api/classes").get_json()["count"] == 6


def test_predict_with_multipart_file(client):
    data = {"file": (io.BytesIO(make_image_bytes("JPEG")), "photo.jfif")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    body = response.get_json()
    assert response.status_code == 200 and body["success"]
    assert len(body["all_probabilities"]) == 6
    assert len(body["top_predictions"]) == 3


def test_predict_with_base64_json(client):
    payload = base64.b64encode(make_image_bytes("PNG")).decode()
    response = client.post("/api/predict", json={"image_base64": "data:image/png;base64," + payload})
    assert response.status_code == 200 and response.get_json()["success"]


def test_predict_rejects_missing_and_corrupt_input(client):
    assert client.post("/api/predict").status_code == 400
    data = {"file": (io.BytesIO(b"not an image"), "x.jpg")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400 and not response.get_json()["success"]


def test_oversized_upload_is_rejected(client):
    response = client.post("/api/predict", data=b"x" * (11 * 1024 * 1024),
                           content_type="application/octet-stream")
    assert response.status_code == 413
