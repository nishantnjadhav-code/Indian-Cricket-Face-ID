from pathlib import Path
import json

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cricket_player_model.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.60


st.set_page_config(
    page_title="Indian Cricket Face Identification",
    page_icon="🏏",
    layout="centered",
)

st.title("🏏 Indian Cricket Team Member Face Identification")

st.write(
    "Upload a photograph to classify it among the Indian cricket "
    "players represented in the training dataset."
)

st.info(
    "This is a research and educational image-classification demo. "
    "Predictions can be incorrect and should not be treated as "
    "verified identity."
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASSES_PATH}"
        )

    # Compatibility workaround for older serialized initializer configs.
    class CompatibleGlorotUniform(keras.initializers.GlorotUniform):
        def __init__(self, seed=None, input_axes=None, **kwargs):
            super().__init__(seed=seed, **kwargs)

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "GlorotUniform": CompatibleGlorotUniform,
            "keras.initializers>GlorotUniform":
                CompatibleGlorotUniform,
        },
    )

    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output size does not match class_names.json."
        )

    return model, class_names


def classify_image(image, model, class_names):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    scores = model.predict(array, verbose=0)[0]

    indices = np.argsort(scores)[::-1][:3]
    best_index = int(indices[0])
    confidence = float(scores[best_index])

    return {
        "top_candidate": class_names[best_index],
        "confidence": confidence,
        "top_three": [
            (class_names[int(i)], float(scores[i]))
            for i in indices
        ],
    }


uploaded_file = st.file_uploader(
    "Choose a player image",
    type=["jpg", "jpeg", "png", "webp"],
)

threshold = st.slider(
    "Minimum model score for a named prediction",
    min_value=0.30,
    max_value=0.95,
    value=UNKNOWN_THRESHOLD,
    step=0.05,
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        with st.spinner("Loading model and analyzing image..."):
            model, class_names = load_artifacts()
            result = classify_image(image, model, class_names)

        if result["confidence"] < threshold:
            st.warning(
                "No named prediction meets the selected score "
                f"threshold. Top candidate: {result['top_candidate']}."
            )
        else:
            st.success(
                f"Predicted player: {result['top_candidate']}"
            )

        st.metric(
            "Top candidate model score",
            f"{result['confidence'] * 100:.2f}%",
        )

        st.caption(
            "This is a softmax score, not a calibrated probability "
            "that the identity is correct."
        )

        st.subheader("Top candidate scores")

        chart_data = {
            name: score
            for name, score in result["top_three"]
        }

        st.bar_chart(chart_data, horizontal=True)

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()

st.subheader("About this project")

st.write(
    "The application uses a MobileNetV2 transfer-learning classifier "
    "trained on labeled images of selected cricket players. "
    "Predictions are limited to the trained classes and may be wrong."
)
