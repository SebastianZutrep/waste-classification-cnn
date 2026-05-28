from keras.models import load_model
import tensorflow as tf
import keras
import numpy as np
import cv2

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)


# Cargar el modelo

model = load_model(
    'Recursos/Model/mejor_modelo.keras' 
)

# Cargar etiquetas (sin números)
with open('Recursos/Model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    

# Preprocesamiento de imagen
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Cargar imágenes de los bins
bins_images = {
    "Residuos aprovechables": cv2.imread('Recursos/Bins/Residuos_aprovechables.png'),
    "Residuos no aprovechables": cv2.imread('Recursos/Bins/Residuos_no_aprovechables.png'),
    "Residuos organicos aprovechables": cv2.imread('Recursos/Bins/Residuos_organicos_aprovechables.png'),
    "Residuos peligrosos": cv2.imread('Recursos/Bins/Residuos_peligrosos.png'),
    "No identificado": cv2.imread('Recursos/Bins/No_identificado.png')
}

# Mapeo de etiquetas a contenedores
bins_map = {
    "Biodegradable": "Residuos organicos aprovechables",
    "Biological Waste": "Residuos peligrosos",
    "Cardboard": "Residuos aprovechables",
    "Food-Contaminated Paper, Cardboard & Napkins": "Residuos no aprovechables",
    "Glass": "Residuos aprovechables",
    "Metal": "Residuos aprovechables",
    "Paper": "Residuos aprovechables", 
    "Plastic": "Residuos aprovechables",
}

# Captura de video
cap = cv2.VideoCapture(0)  # Cambia a 0 si usas la cámara integrada

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("Error: No se pudo capturar imagen de la cámara.")
        continue

    imgResize = cv2.resize(img, (720, 490))
    imgBackground = cv2.imread('Recursos/canva.png')

    # Procesar imagen
    preprocessed_img = preprocess_image(img)

    # Realizar predicción
    prediction = model.predict(preprocessed_img)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_prob = prediction[0][class_idx] * 100

    if class_idx < len(labels):
        label = labels[class_idx]
    else:
        label = "No identificado"
        class_prob = 0

    print(f"Material identificado: {label} ({class_prob:.2f}%)")

    # Obtener nombre del bin
    bin_name = bins_map.get(label, "No identificado")
    print(f"Clasificación: {bin_name}")

    bin_image = bins_images.get(bin_name, bins_images["No identificado"])
    bin_image_resized = cv2.resize(bin_image, (200, 200))

    # Mostrar todo en el background
    imgBackground[200:400, 50:250] = bin_image_resized
    imgBackground[120:610, 515:1235] = imgResize


    # Definir color según precisión
    if class_prob >= 80:
        color = (0, 128, 0)  # Verde oscuro
    elif class_prob >= 50:
        color = (0, 165, 255)  # Naranja
    else:
        color = (0, 0, 255)  # Rojo


    # Mostrar texto en pantalla
    cv2.putText(imgBackground, f"Material: {label}", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(imgBackground, f"Clasificacion: {bin_name}", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(imgBackground, f"Precision: {class_prob:.2f}%", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Mostrar ventanas
    cv2.imshow("Camara", img)
    cv2.imshow("Clasificador de Residuos", imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
