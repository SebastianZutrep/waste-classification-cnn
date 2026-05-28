from utils import load_trained_model, load_labels, preprocess_image
import cv2
import numpy as np

model = load_trained_model()
labels = load_labels()

def predict(image_path):
    img = cv2.imread(image_path)
    img = preprocess_image(img)
    preds = model.predict(img)
    return labels[np.argmax(preds)]


