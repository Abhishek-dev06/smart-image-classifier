image_dataset_from_directory
Train accuracy (best)  : 72.29%
Validation accuracy    : 69.54%
## Failure Analysis

**Image:** <file name>
**Actual class:** <actual>
**Predicted class:** <predicted>
**Confidence:** <xx.xx%>

**What I observed:** <2-3 lines: kya dikhta hai image mein, jaise lighting,
background, object ki position>

**Likely explanation:** A likely explanation is <...>. The misclassification
may have occurred because <...>. I have not verified this by an experiment.

**Supporting evidence:** <jaise "training set mein <actual> ki N images hain
vs <predicted> ki M", ya "confusion matrix mein ye pair sabse common hai">

**Possible improvement:** <jaise more varied examples of <actual>, transfer
learning with MobileNetV2, class balancing>
# Project Decisions

## Dataset Choice

I chose the <dataset name> dataset (<N> classes: <class names>).
Reason: <your own reason, e.g. clear visual classes, provided by GDG-USAR,
easy to explain, small enough to train on a laptop CPU>.

Dataset summary (from `src/dataset_analysis.py`): <total> images, imbalance ratio
<ratio> (largest/smallest class). <Verdict from the facts sheet.>
<Explain the effect in your own words: with imbalance, accuracy can hide weak
performance on the small class, so I also report per-class precision, recall
and F1. I did not apply class weights in the baseline.>

## Train/Validation/Test Split

70% training, 15% validation, 15% test, split per class (so every class appears
in all three sets) with a fixed seed (42). Resulting sizes: <train>/<val>/<test>.
Validation data is used to choose the best epoch and reduce the learning rate.
The test set was never used during training or tuning and was evaluated once
at the end. The split happens before any augmentation, so no augmented copy of
a test image can leak into training.

Known limitation: if the public dataset contains near-duplicate photos, some
may fall into different splits and make results look slightly better.

## Image Size

128 x 128. It keeps enough detail for the classes, trains reasonably fast on a
CPU, and keeps the model small. Resizing stretches images to a square, so the
aspect ratio is not preserved. <224 x 224 would give more detail but is slower.>

## Preprocessing

- Images are converted to RGB and resized to 128 x 128.
- Pixel values are scaled from 0-255 to 0-1 (`Rescaling(1/255)`), which keeps
  inputs in a small range and makes training more stable.
- Normalization is a layer inside the model, so `predict.py` and the Streamlit
  app cannot accidentally use a different scaling than training.

## Augmentation

Random horizontal flip, rotation (10%) and zoom (10%), applied only during
training. <Why each is safe for this dataset: e.g. flipping a bottle or box does
not change its class.> Augmentation layers are automatically inactive during
validation, testing and prediction. <If you used another dataset such as traffic
signs, state that horizontal flip was disabled because it changes the meaning.>

## Model Architecture

A small CNN: three Conv2D + MaxPooling blocks (32, 64, 128 filters), global
average pooling, Dense(128), Dropout(0.3) and a softmax layer. Total parameters:
<count>. I used a small model because the dataset is small and a large model
would overfit more easily. It is also a clear baseline that can be explained
layer by layer. Global average pooling was used instead of Flatten to reduce the
number of parameters in the dense layer.

## Optimizer

Adam with learning rate 0.001. It adapts the step size per parameter and works
well with default settings, which suits a baseline. `ReduceLROnPlateau` halves
the learning rate when validation loss stops improving.

## Loss Function

Sparse categorical cross-entropy, because labels are integers (0 to <N-1>) and
the output layer uses softmax over <N> classes.

## Training

Up to 30 epochs with `EarlyStopping` (patience 6, restores best weights) and
`ModelCheckpoint` on validation loss. Training stopped after <epochs run>
epochs; the best epoch was <best epoch>. Training accuracy <x>% vs validation
accuracy <y>% at the best epoch (gap <z> points).
<Describe what the curves show, honestly: e.g. "validation accuracy stayed close
to training accuracy" or "the gap suggests some overfitting". Remember that
training accuracy is measured with augmentation on.>

## Evaluation

Test accuracy: <x>% on <n> test images. Accuracy alone can be misleading when
classes are imbalanced, so I also report precision, recall and F1 for each
class (macro F1 <value>). The weakest class was <class> (F1 <value>).
The confusion matrix shows where the model mixes classes up. The most common
confusion was <actual> -> <predicted> (<count> images). <Your interpretation,
e.g. "A likely explanation is that these classes look similar in shape or
texture. I did not verify this with an experiment.">

Softmax confidence is not a calibrated probability. A wrong prediction can still
have high confidence.

## Unseen Image Tests

I tested <k> photos that are not in the dataset (`test_images/`). Results:
<one line per image or a short summary, including the wrong ones>.
<Mention that some images were chosen because they were difficult.>

## Failure Analysis

**Image:** <file>
**Actual class:** <actual>
**Predicted class:** <predicted>
**Confidence:** <xx.xx%>

**What I observed:** <what you actually see in the image: lighting, background,
object position, similar-looking class>

**Likely explanation:** A likely explanation is <...>. The misclassification may
have occurred because <...>. I have not confirmed this with a controlled
experiment.

**Supporting evidence:** <e.g. number of training images for both classes from
the facts sheet, or that this pair is the most common confusion in the matrix>

**Possible improvement:** <e.g. more varied examples of the actual class,
transfer learning, class balancing>

## Future Improvements

- Transfer learning with MobileNetV2 or EfficientNet (ImageNet weights)
- More and more varied training data, including hard cases
- Class balancing or class weights
- Hyperparameter tuning, judged on the validation set (not the test set)
- Stronger but still class-safe augmentation
- Better confidence calibration and an "unknown" option for objects outside the classes