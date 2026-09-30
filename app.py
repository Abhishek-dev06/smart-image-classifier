"""Streamlit web interface for Smart Image Classifier."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# app.py lives in the project root, the other modules live in src/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import LOW_CONFIDENCE, load_image, load_model_and_classes, predict  # noqa: E402

MAX_UPLOAD_MB = 10


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Load the model once and reuse it on every rerun."""
    return load_model_and_classes()


def main():
    st.set_page_config(page_title="Smart Image Classifier", page_icon="🖼️")

    st.title("Smart Image Classifier")
    st.write("Upload an image and the CNN will predict its category.")

    try:
        model, class_names = get_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:  # corrupted or incompatible model file
        st.error(f"Could not load the model: {error}")
        st.stop()

    st.caption("Classes: " + ", ".join(class_names))

    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Choose a .jpg, .jpeg or .png image to get started.")
        return

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is too large. Please upload an image under {MAX_UPLOAD_MB} MB.")
        return

    try:
        image = load_image(uploaded)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception:
        st.error("This file could not be opened as an image.")
        return

    with st.spinner("Classifying..."):
        ranked = predict(model, class_names, image)
    label, confidence = ranked[0]

    left, right = st.columns(2)
    with left:
        st.subheader("Preview")
        st.image(image)
    with right:
        st.subheader("Prediction")
        st.markdown(f"## {label.capitalize()}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        if confidence < LOW_CONFIDENCE:
            st.warning("Low confidence. Treat this prediction with caution.")

    st.subheader("Top 3 Predictions")
    for name, probability in ranked[:3]:
        st.progress(probability, text=f"{name.capitalize()}  {probability * 100:.2f}%")

    with st.expander("All class probabilities"):
        table = pd.DataFrame({
            "class": [name for name, _ in ranked],
            "probability (%)": [round(p * 100, 2) for _, p in ranked],
        }).set_index("class")
        st.bar_chart(table)

    st.caption(
        "Confidence is the model's softmax probability. It is not a guaranteed "
        "real-world certainty, and the model only knows the classes listed above."
    )


main()