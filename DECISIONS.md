# Engineering Decisions

## 1. Scope: 12 waste classes
The model predicts 12 classes: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass. It is a closed-set classifier. Unrelated images still get a waste label, and the UI says so. The 0.60 warning is a heuristic, not calibration.

## 2. Backbone and input
MobileNetV2 (ImageNet weights, frozen) + GlobalAveragePooling + Dense(128) + Dropout(0.3) + 12-way softmax. Input is 224 x 224 RGB, bilinear resize, float values 0-255. A `Rescaling(1/255)` layer inside the model normalizes once; preprocessing outside the model must not divide by 255 again.

Open point: MobileNetV2's ImageNet weights were trained on inputs in [-1, 1], while this model feeds [0, 1]. It works, but `Rescaling(1/127.5, offset=-1)` is worth testing in a future run. Retrain and re-evaluate if changed.

## 3. Artifacts must match
`image_classifier.keras`, `class_names.json` and `model_metadata.json` come from one run. `train.py` writes all three. `predict.py` refuses to load if the output count differs from the label count, or if `class_names.json` and `model_metadata.json` disagree on label order.

## 4. Data split
`prepare_dataset.py` validates every class (>= 10 readable images) before deleting old splits, then splits per class 70/15/15 with seed 42. Split is file-based, not grouped by near-duplicates.

## 5. Inference formats
Uploads accept JPEG, JFIF, PNG, WebP, BMP, TIFF. Training accepts JPEG/PNG only. EXIF orientation is applied, transparency is composited on white, oversized or corrupt files give clear errors.

## 6. Results
All reported metrics must come from the 12-class run's `results/*.json`. Older 6-class results were removed.
