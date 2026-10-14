
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cricket_player_model.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.60


def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "'python src/train_model.py' first."
        )

    model = keras.models.load_model(MODEL_PATH)

    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def predict_image(image_path, threshold=UNKNOWN_THRESHOLD):
    model, class_names = load_model_and_classes()

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    if confidence < threshold:
        predicted_name = "Unknown Player"
    else:
        predicted_name = class_names[best_index]

    return {
        "player_name": predicted_name,
        "top_candidate": class_names[best_index],
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "is_unknown": confidence < threshold,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify an Indian cricket player from an image."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=UNKNOWN_THRESHOLD,
        help="Minimum softmax score for a named prediction",
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    result = predict_image(args.image, args.threshold)

    print("\nPrediction result")
    print("-" * 35)
    print(f"Predicted player: {result['player_name']}")
    print(f"Top candidate: {result['top_candidate']}")
    print(f"Model score: {result['confidence_percent']}%")
