import cv2
import numpy as np

IMG_SIZE = (128, 128)

def preprocess_image(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def load_labels(labels_path):
    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]

