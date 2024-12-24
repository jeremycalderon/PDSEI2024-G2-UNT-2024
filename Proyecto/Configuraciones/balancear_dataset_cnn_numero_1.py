# Paso 1: Importar librerías necesarias
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paso 2: Definir las rutas locales
# Cambia estas rutas según la ubicación de tu dataset en tu máquina local
ruta_dataset_original = './garbage-dataset_clean'  # Ruta al dataset original
ruta_dataset_balanceado = './garbage-dataset_undersample'  # Ruta para guardar el dataset balanceado

# Verificar las clases (subcarpetas en el dataset original)
clases = os.listdir(ruta_dataset_original)
print("Clases encontradas:", clases)

# Paso 3: Cargar las imágenes y las etiquetas correspondientes
imagenes = []
etiquetas = []
for idx, clase in enumerate(clases):
    ruta_clase = os.path.join(ruta_dataset_original, clase)
    imagenes_clase = glob(os.path.join(ruta_clase, '*.jpg'))  # Busca imágenes en formato .jpg
    imagenes.extend(imagenes_clase)  # Agrega las rutas de las imágenes
    etiquetas.extend([idx] * len(imagenes_clase))  # Asocia la clase a cada imagen

imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Paso 4: Convertir las imágenes a vectores numéricos usando VGG16
# Cargar el modelo preentrenado VGG16
modelo_base = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Función para convertir una imagen en un vector
def imagen_a_vector(ruta_imagen, idx_imagen, total_imagenes):
    img = load_img(ruta_imagen, target_size=(224, 224))  # Redimensiona la imagen a 224x224
    img_array = img_to_array(img)  # Convierte la imagen a un array
    img_array = np.expand_dims(img_array, axis=0)  # Agrega una dimensión extra para batch
    img_array = preprocess_input(img_array)  # Preprocesa la imagen para VGG16
    # Muestra el número de imagen que se está procesando
    print(f"Imagen {idx_imagen + 1}/{total_imagenes}")
    return modelo_base.predict(img_array).flatten()  # Devuelve el vector de características

# Convertir todas las imágenes a vectores
print("\nExtrayendo características de las imágenes...")
vectores = np.array([imagen_a_vector(img, idx, len(imagenes)) for idx, img in enumerate(imagenes)])

# Paso 5: Aplicar NearMiss para balancear las clases
nearmiss = NearMiss()
vectores_balanceados, etiquetas_balanceadas = nearmiss.fit_resample(vectores, etiquetas)

print("\nDistribución después del undersampling:")
from collections import Counter
print(Counter(etiquetas_balanceadas))

# Paso 6: Seleccionar las imágenes correspondientes
imagenes_balanceadas = imagenes[nearmiss.sample_indices_]

# Paso 7: Guardar las imágenes balanceadas en el nuevo dataset
os.makedirs(ruta_dataset_balanceado, exist_ok=True)  # Crear la carpeta principal

for idx, clase in enumerate(clases):
    # Crear subcarpeta para cada clase
    ruta_clase_balanceada = os.path.join(ruta_dataset_balanceado, clase)
    os.makedirs(ruta_clase_balanceada, exist_ok=True)

    # Copiar las imágenes de la clase balanceada
    for img in imagenes_balanceadas[etiquetas_balanceadas == idx]:
        shutil.copy(img, ruta_clase_balanceada)  # Copia las imágenes seleccionadas

print("\nEl dataset balanceado se ha creado en:", ruta_dataset_balanceado)
