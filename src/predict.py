import tensorflow as tf
import numpy as np
from utils import preprocess_image, load_labels

MODEL_PATH = "model/mejor_modelo.keras"
LABELS_PATH = "model/labels.txt"

def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)

    img = preprocess_image(image_path)
    prediction = model.predict(img)

    return labels[np.argmax(prediction)]

if __name__ == "__main__":
    result = predict("example.jpg")
    print("Predicted waste type:", result)

