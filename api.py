"""REST API server for WasteWealth MobileNetV2 Image Classifier."""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils import (CLASS_NAMES_PATH, INFERENCE_EXTENSIONS, MODEL_METADATA_PATH,
                   MODEL_PATH)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Flutter, Web, Android, etc.

# Lazy-loaded model and classes
_model = None
_class_names = None
_metadata = None


def get_metadata():
    global _metadata
    if _metadata is None and MODEL_METADATA_PATH.is_file():
        try:
            with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
                _metadata = json.load(f)
        except Exception:
            _metadata = {}
    return _metadata or {}


def get_classes():
    global _class_names
    if _class_names is None:
        if CLASS_NAMES_PATH.is_file():
            with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
                _class_names = json.load(f)
        else:
            _class_names = [
                "battery", "biological", "brown-glass", "cardboard", "clothes",
                "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
            ]
    return _class_names


def get_model():
    global _model, _class_names
    if _model is None:
        from predict import load_model_and_classes
        _model, _class_names = load_model_and_classes()
    return _model, _class_names


HTML_TESTER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WasteWealth MobileNetV2 - REST API</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
    .container { max-width: 800px; margin: auto; background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    h1 { color: #38bdf8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem; }
    .badge { background: #0284c7; color: white; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.85rem; font-weight: 500; }
    .endpoints { background: #0f172a; padding: 1rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid #334155; }
    .endpoint { margin-bottom: 0.5rem; font-family: monospace; font-size: 0.95rem; }
    .method { background: #10b981; color: black; font-weight: bold; padding: 0.15rem 0.4rem; border-radius: 4px; margin-right: 0.5rem; }
    .method.post { background: #f59e0b; }
    .drop-zone { border: 2px dashed #475569; padding: 2rem; text-align: center; border-radius: 8px; margin: 1.5rem 0; cursor: pointer; transition: 0.2s; }
    .drop-zone:hover { border-color: #38bdf8; background: #1e293b11; }
    input[type=file] { display: none; }
    button { background: #0284c7; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; width: 100%; transition: 0.2s; }
    button:hover { background: #0369a1; }
    #result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; background: #0f172a; border: 1px solid #334155; display: none; white-space: pre-wrap; font-family: monospace; }
    #preview { max-width: 200px; max-height: 200px; margin: 1rem auto; display: none; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>♻️ WasteWealth MobileNetV2 <span class="badge">REST API Active</span></h1>
    <p>Predict material waste category across 12 classes using our high-performance API endpoint.</p>
    
    <div class="endpoints">
      <div class="endpoint"><span class="method">GET</span> <code>/api/health</code> - API Status check</div>
      <div class="endpoint"><span class="method">GET</span> <code>/api/classes</code> - List all 12 supported classes</div>
      <div class="endpoint"><span class="method">GET</span> <code>/api/model</code> - Model specifications & metadata</div>
      <div class="endpoint"><span class="method post">POST</span> <code>/api/predict</code> - Send image (multipart/form-data: <code>file</code> or JSON base64)</div>
    </div>

    <h3>Interactive API Tester</h3>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <p id="dropText">📁 Click or drop a photo here to test the API</p>
      <img id="preview" alt="Preview" />
      <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])" />
    </div>

    <button id="submitBtn" onclick="uploadImage()" disabled>Select an image to test API</button>
    <div id="result"></div>
  </div>

  <script>
    let selectedFile = null;
    function handleFile(file) {
      if (!file) return;
      selectedFile = file;
      document.getElementById('dropText').innerText = 'Selected: ' + file.name;
      const preview = document.getElementById('preview');
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
      const btn = document.getElementById('submitBtn');
      btn.disabled = false;
      btn.innerText = 'Send POST /api/predict';
    }

    async function uploadImage() {
      if (!selectedFile) return;
      const btn = document.getElementById('submitBtn');
      const resultDiv = document.getElementById('result');
      btn.innerText = 'Calling API...';
      btn.disabled = true;
      resultDiv.style.display = 'block';
      resultDiv.innerText = 'Processing...';

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();
        resultDiv.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        resultDiv.innerText = 'API Error: ' + err;
      } finally {
        btn.innerText = 'Send POST /api/predict';
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def home():
    """Home route providing interactive API UI and documentation."""
    return render_template_string(HTML_TESTER)


@app.route("/api/health", methods=["GET"])
def health():
    """Check API and model readiness."""
    meta = get_metadata()
    return jsonify({
        "status": "healthy",
        "model_name": meta.get("model_name", "WasteWealth MobileNetV2"),
        "number_of_classes": len(get_classes()),
        "model_file_exists": MODEL_PATH.is_file(),
    })


@app.route("/api/classes", methods=["GET"])
def classes():
    """Return the 12 supported waste classes."""
    cls = get_classes()
    return jsonify({
        "classes": cls,
        "count": len(cls)
    })


@app.route("/api/model", methods=["GET"])
def model_info():
    """Return model specifications & metadata."""
    meta = get_metadata()
    return jsonify(meta)


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """Run inference on an uploaded photo.
    
    Accepts:
    - Multipart file field: 'file' or 'image'
    - JSON field: 'image_base64'
    """
    try:
        from predict import load_image, predict
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"Dependency error: {e}. Please ensure requirements.txt is installed."
        }), 500

    image_source = None

    if "file" in request.files and request.files["file"].filename:
        image_source = request.files["file"]
    elif "image" in request.files and request.files["image"].filename:
        image_source = request.files["image"]
    elif request.is_json and "image_base64" in request.json:
        try:
            b64_data = request.json["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            image_source = io.BytesIO(image_bytes)
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid base64 payload: {e}"}), 400
    elif request.data:
        image_source = io.BytesIO(request.data)
    else:
        return jsonify({
            "success": False,
            "error": "No image provided. Send a file via 'file' or JSON 'image_base64'."
        }), 400

    try:
        img = load_image(image_source)
    except Exception as e:
        return jsonify({"success": False, "error": f"Could not decode image: {e}"}), 400

    try:
        model, class_names = get_model()
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Model error: {e}"
        }), 500

    try:
        ranked = predict(model, class_names, img)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Classification failed: {e}"
        }), 500

    top_class, top_conf = ranked[0]
    meta = get_metadata()

    return jsonify({
        "success": True,
        "model": meta.get("model_name", "WasteWealth MobileNetV2"),
        "predicted_class": top_class,
        "confidence": round(float(top_conf), 4),
        "top_predictions": [
            {"class": name, "probability": round(float(prob), 4)}
            for name, prob in ranked[:3]
        ],
        "all_probabilities": {
            name: round(float(prob), 4) for name, prob in ranked
        }
    })


def main():
    parser = argparse.ArgumentParser(description="Run WasteWealth MobileNetV2 REST API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    meta = get_metadata()
    model_name = meta.get("model_name", "WasteWealth MobileNetV2")
    print("=" * 60)
    print(f"Starting {model_name} REST API Server")
    print(f"Local URL   : http://localhost:{args.port}")
    print(f"Network URL : http://{args.host}:{args.port}")
    print("Interactive Web Tester & Docs available at the root URL")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

