import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys


# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 25

# Rutas
waste_dir = r"C:\Users\SEBASTIAN\Desktop\Clasificacion de residuos IA\Recursos\Waste"
mask_dir = r"C:\Users\SEBASTIAN\Desktop\Clasificacion de residuos IA\Recursos\Masks"

# Clases
class_names = sorted(next(os.walk(waste_dir))[1])
label_dict = {cls: idx for idx, cls in enumerate(class_names)}

# Data Augmentation (solo se aplica al generador de entrenamiento)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1)
])


# Generador personalizado basado en ImageDataGenerator
class MaskImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, waste_dir, mask_dir, class_names, label_dict, batch_size=16, img_size=(128, 128), shuffle_data=True, subset='train', validation_split=0.2):
        self.waste_dir = waste_dir
        self.mask_dir = mask_dir
        self.class_names = class_names
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle_data = shuffle_data
        self.subset = subset
        self.validation_split = validation_split

        self.samples = self._collect_samples()

        if self.shuffle_data:
            self.samples = shuffle(self.samples, random_state=42)

        split_idx = int(len(self.samples) * (1 - self.validation_split))
        if self.subset == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def _collect_samples(self):
        samples = []
        for class_name in self.class_names:
            waste_class_dir = os.path.join(self.waste_dir, class_name)
            mask_class_dir = os.path.join(self.mask_dir, class_name)

            for root, _, files in os.walk(waste_class_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, file)
                        rel_path = os.path.relpath(img_path, waste_class_dir)
                        mask_path = os.path.join(mask_class_dir, rel_path)
                        if os.path.exists(mask_path):
                            samples.append((img_path, mask_path, self.label_dict[class_name]))
        return samples

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        batch = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []
        for img_path, mask_path, label in batch:
            try:
                img = load_img(img_path, target_size=self.img_size)
                mask = load_img(mask_path, color_mode='grayscale', target_size=self.img_size)

                img = img_to_array(img) / 255.0
                mask = img_to_array(mask) / 255.0
                if mask.shape[-1] != 1:
                    mask = mask[..., :1]

                masked_img = img * mask
                if self.subset == 'train':
                    masked_img = data_augmentation(masked_img, training=True)

                X.append(masked_img)
                y.append(label)
            except Exception as e:
                print(f"[ERROR] Falló el procesamiento de: {img_path}. Error: {e}")

        return np.array(X), to_categorical(y, num_classes=len(self.class_names))

    def on_epoch_end(self):
        if self.shuffle_data:
            self.samples = shuffle(self.samples, random_state=42)

# Crear generadores
train_gen = MaskImageDataGenerator(waste_dir, mask_dir, class_names, label_dict, batch_size=BATCH_SIZE, img_size=IMG_SIZE, subset='train')
val_gen = MaskImageDataGenerator(waste_dir, mask_dir, class_names, label_dict, batch_size=BATCH_SIZE, img_size=IMG_SIZE, subset='validation')

# Modelo con DenseNet121
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(
    "mejor_modelo.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Entrenamiento con EarlyStopping y Checkpoint
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# Guardar modelo final (opcional, ya se guardó el mejor automáticamente con el checkpoint)
model.save("modelo_residuos_con_densenet_mascaras.keras")

# Obtener historial
history = history.history


# Gráfica de precisión (accuracy)
plt.figure(figsize=(10, 5))
plt.plot(history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.savefig("precision_modelo.jpg", dpi=300)
plt.show()

# Gráfica de pérdida (loss)
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Pérdida Entrenamiento')
plt.plot(history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig("perdida_modelo.jpg", dpi=300)
plt.show()

# Obtener predicciones y etiquetas reales
y_true = []
y_pred = []

for batch_imgs, batch_labels in val_gen:
    preds = model.predict(batch_imgs)
    y_true.extend(np.argmax(batch_labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Graficar matriz de confusión
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión")
plt.grid(False)
plt.savefig("matriz_confusion.jpg", dpi=300)
plt.show()

# Finaliza el programa
sys.exit()
