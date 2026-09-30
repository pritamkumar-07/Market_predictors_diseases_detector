from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "disease_dataset" / "train"
VAL_DIR = ROOT / "disease_dataset" / "validation"
TEST_DIR = ROOT / "disease_dataset" / "test"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 5

if not TRAIN_DIR.exists():
    raise SystemExit("Create disease_dataset/train/<class_name>/ and add real labeled images.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Rescaling(1.0 / 255.0),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),

    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.30),
    \
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(MODEL_DIR / "disease_cnn.keras")
(MODEL_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))

if TEST_DIR.exists():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

print("Disease CNN saved to:", MODEL_DIR / "disease_cnn.keras")
print("Classes:", class_names)
