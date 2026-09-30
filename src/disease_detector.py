import numpy as np
from PIL import Image
import tensorflow as tf


IMAGE_SIZE = (96, 96)


def load_disease_model(path):
    return tf.keras.models.load_model(path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_disease(model, image: Image.Image, class_names):
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])
