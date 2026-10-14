import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "dataset" / "processed" / "test"
MODEL_PATH = ROOT / "models" / "cricket_player_model.keras"
CLASSES_PATH = ROOT / "models" / "class_names.json"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading class names...", flush=True)
    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    print("Loading model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test dataset...", flush=True)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )

    if test_ds.class_names != class_names:
        raise RuntimeError(
            "Test class ordering differs from training class ordering."
        )

    y_true = []
    y_pred = []

    print("Predicting in small batches...", flush=True)

    for images, labels in test_ds:
        probabilities = model(images, training=False).numpy()
        predictions = np.argmax(probabilities, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

        del probabilities, predictions, images, labels

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    report_text = (
        f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
        f"Test images: {len(y_true)}\n"
        f"Correct predictions: {int(np.sum(y_true == y_pred))}\n"
        f"Incorrect predictions: {int(np.sum(y_true != y_pred))}\n\n"
        f"Classification report:\n{report}\n"
    )

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\nEvaluation completed!", flush=True)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Test images: {len(y_true)}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
