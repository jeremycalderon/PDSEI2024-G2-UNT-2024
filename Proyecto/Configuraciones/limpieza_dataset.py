import shutil
import os
import math

#Eliminar carpetas
# carpeta = 'garbage-dataset/biological'

# try:
#     shutil.rmtree(carpeta)
#     print(f'Carpeta "{carpeta}" eliminada junto con su contenido')
    
# except FileNotFoundError:
#     print(f'La carpeta "{carpeta}" no existe')

# except OSError as e:
#     print(f'Error al eliminar la carpeta: {e}')

#Eliminar imágenes dentro de las carpetas seleccionadas
# carpeta2 = 'garbage-dataset/paper'
# nombre_eliminar = 'paper_686'

extensiones = ['.jpg']

# try:
#     for archivo in os.listdir(carpeta2):
#         if archivo.startswith(nombre_eliminar) and archivo.lower().endswith(tuple(extensiones)):
#             ruta_completa = os.path.join(carpeta2, archivo)
#             os.remove(ruta_completa)
#             print(f'Eliminado: {archivo}')
#     print('Archivo eliminado')

# except FileNotFoundError:
#     print('La carpeta no existe')

# except Exception as e:
#     print(f'Ocurrió un error: {e}')

#Mover imágenes de una carpeta a otra
# carpeta_origen = 'garbage-dataset/paper'
# carpeta_destino = 'garbage-dataset/cardboard'
# nombre_cambiar = 'paper_1009'

# extensiones = ['.jpg','.png']

# os.makedirs(carpeta_destino, exist_ok=True)

# try:
#     for archivo in os.listdir(carpeta_origen):
#         if archivo.startswith(nombre_cambiar) and archivo.lower().endswith(tuple(extensiones)):
#             ruta_origen = os.path.join(carpeta_origen, archivo)
#             ruta_destino = os.path.join(carpeta_destino, archivo)
#             shutil.move(ruta_origen, ruta_destino)
#             print(f'Moviendo: {archivo}')
#     print('Archivo desplazado')

# except FileNotFoundError:
#     print('La carpeta de origen no existe')

# except Exception as e:
#     print(f'Ocurrió un error: {e}')

#Renombrar imágenes de una carpeta
# carpeta3 = 'garbage-dataset/battery'
# tipo = 'battery'

# try:
#     for i, archivo in enumerate(os.listdir(carpeta3), start=1):
#         if archivo.lower().endswith(tuple(extensiones)):
#             ruta_inicial = os.path.join(carpeta3, archivo)
#             nuevo_nombre = f"{tipo}_{i}{os.path.splitext(archivo)[1]}"
#             ruta_final = os.path.join(carpeta3, nuevo_nombre)
#             os.rename(ruta_inicial, ruta_final)
#             print(f'Renombrando: {archivo} -> {nuevo_nombre}')
#     print('Archivo renombrado')

# except FileNotFoundError:
#     print('La carpeta no existe')

# except Exception as e:
#     print(f'Ocurrió un error: {e}')

#Creando las carpetas de entrenamiento, validación y prueba
carpeta_inicio = 'garbage-dataset/plastic'
carpeta_destino = [
    'garbage-dataset/entrenamiento',
    'garbage-dataset/validacion',
    'garbage-dataset/prueba'
]

porcentajes = [0.7,0.2,0.1]

for carpeta in carpeta_destino:
    os.makedirs(carpeta, exist_ok=True)

try:
    imagenes = [f for f in os.listdir(carpeta_inicio) if f.lower().endswith(tuple(extensiones))]
    num_imagenes = len(imagenes)
    tamaños = [math.ceil(num_imagenes*p) for p in porcentajes]
    tamaños[-1] = num_imagenes-sum(tamaños[:-1])

    indice = 0

    for i, tamaño in enumerate(tamaños):
        for _ in range(tamaño):
            if indice >= num_imagenes:
                break
            imagen = imagenes[indice]
            ruta_inicio = os.path.join(carpeta_inicio, imagen)
            ruta_destino = os.path.join(carpeta_destino[i], imagen)
            shutil.move(ruta_inicio, ruta_destino)
            print(f'Moviendo: {imagen} -> {carpeta_destino[i]}')
            indice += 1
    print('Todas las imagenes han sido distribuidas')

except FileNotFoundError:
    print('La carpeta de inicio no existe')

except Exception as e:
    print(f'Ocurrió un error: {e}')
