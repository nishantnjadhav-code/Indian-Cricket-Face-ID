
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "cricket_player_model.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
SEED = 42


def load_dataset(directory, shuffle):
    return keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED,
    )


def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Prepared data not found. Run "
            "'python src/prepare_dataset.py' first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset(train_dir, shuffle=True)
    val_ds = load_dataset(val_dir, shuffle=False)

    class_names = train_ds.class_names

    if len(class_names) < 2:
        raise RuntimeError("At least two player classes are required.")

    if val_ds.class_names != class_names:
        raise RuntimeError(
            "Training and validation class folders do not match."
        )

    with CLASSES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        len(class_names),
        activation="softmax",
        name="player_probabilities",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("Starting transfer-learning training...")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    # Fine-tune only the upper layers of MobileNetV2.
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep batch-normalization layers frozen for stability.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    completed_epochs = len(history1.epoch)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=completed_epochs,
        epochs=completed_epochs + FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    # Plot both training stages.
    train_accuracy = (
        history1.history["accuracy"]
        + history2.history["accuracy"]
    )
    val_accuracy = (
        history1.history["val_accuracy"]
        + history2.history["val_accuracy"]
    )
    train_loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_accuracy, label="Training accuracy")
    axes[0].plot(val_accuracy, label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_loss, label="Training loss")
    axes[1].plot(val_loss, label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)

    print("\nTraining completed.")
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASSES_PATH}")
    print(f"Training graph: {OUTPUT_DIR / 'training_history.png'}")
    print("Class names:", class_names)


if __name__ == "__main__":
    main()
