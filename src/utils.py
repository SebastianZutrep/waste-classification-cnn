import cv2
import numpy as np
import tensorflow as tf

# ===============================
# Paths
# ===============================
MODEL_PATH = "Recursos/Model/mejor_modelo.keras"
LABELS_PATH = "Recursos/Model/labels.txt"
BINS_PATH = "Recursos/Bins"

# ===============================
# Load model
# ===============================
def load_trained_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

# ===============================
# Load labels
# ===============================
def load_labels(labels_path=LABELS_PATH):
    with open(labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# ===============================
# Image preprocessing
# ===============================
def preprocess_image(img, img_size=(128, 128)):
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ===============================
# Bins mapping
# ===============================
def get_bins_map():
    return {
        "Biodegradable": "Residuos organicos aprovechables",
        "Biological Waste": "Residuos peligrosos",
        "Cardboard": "Residuos aprovechables",
        "Food-Contaminated Paper, Cardboard & Napkins": "Residuos no aprovechables",
        "Glass": "Residuos aprovechables",
        "Metal": "Residuos aprovechables",
        "Paper": "Residuos aprovechables",
        "Plastic": "Residuos aprovechables",
    }

# ===============================
# Load bin images
# ===============================
def load_bin_images():
    return {
        "Residuos aprovechables": cv2.imread(f"{BINS_PATH}/Residuos_aprovechables.png"),
        "Residuos no aprovechables": cv2.imread(f"{BINS_PATH}/Residuos_no_aprovechables.png"),
        "Residuos organicos aprovechables": cv2.imread(f"{BINS_PATH}/Residuos_organicos_aprovechables.png"),
        "Residuos peligrosos": cv2.imread(f"{BINS_PATH}/Residuos_peligrosos.png"),
        "No identificado": cv2.imread(f"{BINS_PATH}/No_identificado.png"),
    }


