"""
EcoScan — Servidor de clasificación de residuos
Uso: python server.py
Requiere: pip install fastapi uvicorn tensorflow keras numpy opencv-python pillow
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from keras.models import load_model
# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "Recursos" / "Model" / "mejor_modelo.keras"
LABELS_PATH = BASE_DIR / "Recursos" / "Model" / "labels.txt"

# ── Configuración ────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)

BINS_MAP = {
    "Biodegradable":                                  "Residuos organicos aprovechables",
    "Biological Waste":                               "Residuos peligrosos",
    "Cardboard":                                      "Residuos aprovechables",
    "Food-Contaminated Paper, Cardboard & Napkins":   "Residuos no aprovechables",
    "Glass":                                          "Residuos aprovechables",
    "Metal":                                          "Residuos aprovechables",
    "Paper":                                          "Residuos aprovechables",
    "Plastic":                                        "Residuos aprovechables",
}

# ── Carga del modelo (una sola vez al arrancar) ──────────────────────────────
print(f"[EcoScan] TensorFlow {tf.__version__}")
print(f"[EcoScan] Cargando modelo desde {MODEL_PATH} ...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

model = load_model(str(MODEL_PATH))

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f if line.strip()]

print(f"[EcoScan] Modelo cargado. Clases: {labels}")

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="EcoScan API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://waste-classification-cnn.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image: str  # base64 JPEG/PNG


class PredictResponse(BaseModel):
    material: str
    bin: str
    confidence: float
    all_scores: Dict[str, float]


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes de imagen y preprocesa para el modelo."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback con PIL (útil si el navegador envía PNG con alpha)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)          # (1, 128, 128, 3)


@app.get("/")
def root():
    return {"status": "ok", "model": str(MODEL_PATH.name), "classes": labels}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Decodificar base64
    try:
        # Soporta "data:image/jpeg;base64,..." o base64 puro
        b64 = req.image.split(",")[-1]
        image_bytes = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 inválido: {e}")

    # Preprocesar
    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")

    # Inferencia
    preds       = model.predict(tensor, verbose=0)[0]          # (n_classes,)
    class_idx   = int(np.argmax(preds))
    confidence  = float(preds[class_idx]) * 100

    material    = labels[class_idx] if class_idx < len(labels) else "Unknown"
    bin_name    = BINS_MAP.get(material, "No identificado")
    all_scores  = {labels[i]: round(float(preds[i]) * 100, 2) for i in range(len(labels))}

    return PredictResponse(
        material=material,
        bin=bin_name,
        confidence=round(confidence, 2),
        all_scores=all_scores,
    )


# ── Punto de entrada ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)