# Paso 1: Importar librerías necesarias
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
from glob import glob
import numpy as np
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter

# Paso 2: Definir las rutas locales
ruta_dataset_original = './garbage-dataset_clean'  # Ruta al dataset original
ruta_dataset_balanceado = './garbage-dataset_balanceado'  # Ruta para guardar el dataset balanceado

# Verificar las clases (subcarpetas en el dataset original)
clases = os.listdir(ruta_dataset_original)
print("Clases encontradas:", clases)

# Paso 3: Cargar las imágenes y las etiquetas correspondientes
imagenes = []
etiquetas = []
for idx, clase in enumerate(clases):
    ruta_clase = os.path.join(ruta_dataset_original, clase)
    imagenes_clase = glob(os.path.join(ruta_clase, '*.jpg'))
    imagenes.extend(imagenes_clase)
    etiquetas.extend([idx] * len(imagenes_clase))

imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Paso 4: Convertir las imágenes a vectores numéricos usando VGG16
modelo_base = VGG16(weights='imagenet', include_top=False, pooling='avg')

def imagen_a_vector(ruta_imagen, idx_imagen, total_imagenes):
    img = load_img(ruta_imagen, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print(f"Procesando imagen {idx_imagen + 1}/{total_imagenes}")
    return modelo_base.predict(img_array).flatten()

print("\nExtrayendo características de las imágenes...")
vectores = np.array([imagen_a_vector(img, idx, len(imagenes)) for idx, img in enumerate(imagenes)])

# Paso 5: Calcular el tamaño objetivo intermedio
conteo_clases = Counter(etiquetas)
#min_clase = min(conteo_clases.values())
#max_clase = max(conteo_clases.values())
objetivo_intermedio = 1400   #(min_clase + max_clase) // 2  # Tamaño intermedio por clase
print(f"\nEl tamaño objetivo por clase es: {objetivo_intermedio}")

# Paso 6: Reducir clases mayores que el objetivo usando undersampling
undersampler = RandomUnderSampler(sampling_strategy={k: objetivo_intermedio for k, v in conteo_clases.items() if v > objetivo_intermedio})
vectores_undersampled, etiquetas_undersampled = undersampler.fit_resample(vectores, etiquetas)

# Paso 7: Aumentar clases menores que el objetivo usando oversampling (SMOTE)
oversampler = SMOTE(sampling_strategy={k: objetivo_intermedio for k, v in Counter(etiquetas_undersampled).items() if v < objetivo_intermedio})
vectores_balanceados, etiquetas_balanceados = oversampler.fit_resample(vectores_undersampled, etiquetas_undersampled)

print("\nDistribución final después de combinar undersampling y oversampling:")
print(Counter(etiquetas_balanceados))

# Paso 8: Asociar imágenes reales a los vectores seleccionados
indices_reales = undersampler.sample_indices_  # Índices de imágenes reales tras undersampling
imagenes_reales = imagenes[indices_reales]

# Asociar imágenes balanceadas (distinguir reales y sintéticas)
imagenes_balanceadas = []
for idx, etiqueta in enumerate(etiquetas_balanceados):
    if idx < len(imagenes_reales):
        # Imágenes reales provenientes de undersampling
        imagenes_balanceadas.append(imagenes_reales[idx])
    else:
        # Imágenes sintéticas (no asociadas a ninguna imagen real)
        imagenes_balanceadas.append(None)

# Paso 9: Guardar las imágenes reales en el nuevo dataset
os.makedirs(ruta_dataset_balanceado, exist_ok=True)

for idx, clase in enumerate(clases):
    ruta_clase_balanceada = os.path.join(ruta_dataset_balanceado, clase)
    os.makedirs(ruta_clase_balanceada, exist_ok=True)

    for img, etiqueta in zip(imagenes_balanceadas, etiquetas_balanceados):
        if etiqueta == idx and img is not None:
            shutil.copy(img, ruta_clase_balanceada)

print("\nEl dataset balanceado se ha creado en:", ruta_dataset_balanceado)
