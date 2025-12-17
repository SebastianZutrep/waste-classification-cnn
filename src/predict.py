import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH = "model/model.keras"
LABELS = ["Plastic", "Paper", "Glass", "Metal", "Biodegradable"]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return LABELS[np.argmax(prediction)]

if __name__ == "__main__":
    result = predict("example.jpg")
    print("Predicted waste type:", result)
