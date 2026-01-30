import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import urllib.request
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def descomprimir_zip(ruta_zip, carpeta_salida):
    """
    Descomprime un archivo ZIP en la carpeta especificada.
    """
    print(f"Descomprimiendo {ruta_zip}...")
    with zipfile.ZipFile(ruta_zip, "r") as zipf:
        zipf.extractall(carpeta_salida)
    print(f"✓ Descomprimido en: {carpeta_salida}")


def cargar_imagenes_desde_carpetas(ruta_dataset):
    """
    Carga todas las imágenes desde carpetas organizadas por clases.
    
    Args:
        ruta_dataset: Ruta al dataset con estructura carpetas/clases
        
    Returns:
        tuple: (rutas_imagenes, etiquetas, nombres_clases)
            - rutas_imagenes: Lista de rutas a las imágenes
            - etiquetas: Lista de nombres de clases (strings)
            - nombres_clases: Lista de nombres únicos de clases
    """
    rutas_imagenes = []
    etiquetas = []
    
    # Verificar si hay una carpeta extra nivel intermedio
    contenido = os.listdir(ruta_dataset)
    if len(contenido) == 1 and os.path.isdir(os.path.join(ruta_dataset, contenido[0])):
        # Hay una carpeta extra, entrar en ella
        ruta_dataset = os.path.join(ruta_dataset, contenido[0])
        print(f"Detectada carpeta intermedia, usando: {ruta_dataset}")
    
    clases = sorted([d for d in os.listdir(ruta_dataset) 
                     if os.path.isdir(os.path.join(ruta_dataset, d))])
    
    if not clases:
        raise ValueError(f"No se encontraron carpetas de clases en {ruta_dataset}")
    
    print(f"\nClases detectadas: {clases}")
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_dataset, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Cargando clase '{clase}': {len(archivos)} imágenes")
        
        for archivo in archivos:
            ruta_img = os.path.join(ruta_clase, archivo)
            rutas_imagenes.append(ruta_img)
            etiquetas.append(clase)
    
    print(f"\n✓ Total imágenes encontradas: {len(rutas_imagenes)}")
    
    if len(rutas_imagenes) == 0:
        raise ValueError("No se encontraron imágenes. Verifica la estructura del dataset.")
    
    return rutas_imagenes, np.array(etiquetas), clases


def crear_extractor_vgg16():
    """
    Crea el modelo VGG16 para extracción de características usando TensorFlow/Keras.
    
    Returns:
        model: Modelo VGG16 cargado
    """
    print("\nCargando modelo VGG16 con TensorFlow/Keras...")
    
    # Cargar VGG16 preentrenado sin las capas de clasificación
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print("✓ Modelo VGG16 cargado exitosamente")
    
    return model


def preprocesar_imagen(ruta_img):
    """
    Preprocesa una imagen para VGG16 usando Keras.
    
    Args:
        ruta_img: Ruta a la imagen
        
    Returns:
        img: Imagen preprocesada con dimensión de batch
    """
    # Cargar imagen
    img = cv2.imread(ruta_img)
    
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_img}")
    
    # Convertir de BGR a RGB (Keras espera RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar a 224x224
    img = cv2.resize(img, (224, 224))
    
    # Preprocesar con Keras (normalización y mean subtraction)
    img = preprocess_input(img)
    
    # Agregar dimensión de batch
    img = np.expand_dims(img, axis=0)
    
    return img


def extraer_caracteristicas_cnn(rutas_imagenes, model):
    """
    Extrae características CNN de todas las imágenes usando VGG16 con TensorFlow.
    Extrae características de la última capa convolucional (block5_pool) 
    antes de la clasificación - solo FEATURE LEARNING.
    
    Args:
        rutas_imagenes: Lista de rutas a las imágenes
        model: Modelo VGG16
        
    Returns:
        array: Características extraídas aplanadas
    """
    print(f"\n{'='*60}")
    print(f"Extrayendo características CNN usando VGG16 con TensorFlow")
    print(f"Solo capas convolucionales - SIN clasificación")
    print(f"{'='*60}")
    
    caracteristicas = []
    total = len(rutas_imagenes)
    
    for i, ruta in enumerate(rutas_imagenes):
        try:
            # Preprocesar imagen
            img = preprocesar_imagen(ruta)
            
            # Extraer características (block5_pool: 7x7x512 = 25088 features)
            features = model.predict(img, verbose=0)
            
            # Aplanar (convertir de [1, 7, 7, 512] a vector 1D)
            features_flat = features.flatten()
            caracteristicas.append(features_flat)
            
        except Exception as e:
            print(f"⚠ Error procesando {ruta}: {e}")
            # Agregar vector de ceros en caso de error (7x7x512 = 25088)
            caracteristicas.append(np.zeros(25088))
        
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Procesadas {i + 1}/{total} imágenes...")
    
    caracteristicas = np.array(caracteristicas)
    print(f"✓ Completado: {len(caracteristicas)} vectores de características")
    print(f"Dimensión de cada vector: {caracteristicas.shape[1]} (7x7x512 features convolucionales)")
    
    return caracteristicas


def guardar_caracteristicas(caracteristicas, etiquetas, nombres_clases, nombre_base):
    """
    Guarda las características CNN en formato CSV.
    
    Args:
        caracteristicas: Array de características
        etiquetas: Array de etiquetas (nombres de clases)
        nombres_clases: Lista de nombres únicos de clases
        nombre_base: Nombre base del archivo
    """
    if len(caracteristicas) == 0:
        print(f"⚠ No hay características para guardar")
        return
    
    # Guardar en el mismo directorio donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archivo_csv = os.path.join(script_dir, "..", "features", f"{nombre_base}_cnn.csv")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(archivo_csv), exist_ok=True)
    
    # Guardar como CSV
    print(f"\nGuardando características en CSV: {archivo_csv}")
    n_features = caracteristicas.shape[1]
    columnas = [f"feature_{i}" for i in range(n_features)]
    
    df = pd.DataFrame(caracteristicas, columns=columnas)
    df["clase"] = etiquetas
    df.to_csv(archivo_csv, index=False)
    print(f"✓ Guardado CSV: {archivo_csv}")
    print(f"Forma del archivo: {df.shape}")


def procesar_dataset_completo(ruta_zip, nombre_base):
    """
    Pipeline completo: descomprime, carga imágenes y extrae características CNN.
    
    Args:
        ruta_zip: Ruta al archivo ZIP con imágenes preprocesadas
        nombre_base: Nombre base para los archivos de salida
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO DATASET: {nombre_base}")
    print(f"{'='*70}")
    
    # 1. Crear extractor de características
    model = crear_extractor_vgg16()
    
    # 2. Descomprimir
    carpeta_temp = f"temp_{nombre_base}_cnn"
    descomprimir_zip(ruta_zip, carpeta_temp)
    
    # 3. Cargar rutas de imágenes
    rutas_imagenes, etiquetas, nombres_clases = cargar_imagenes_desde_carpetas(carpeta_temp)
    
    # 4. Extraer características CNN
    caracteristicas = extraer_caracteristicas_cnn(rutas_imagenes, model)
    
    # 5. Guardar características
    guardar_caracteristicas(caracteristicas, etiquetas, nombres_clases, nombre_base)
    
    # 6. Limpiar carpeta temporal
    print(f"\nLimpiando carpeta temporal: {carpeta_temp}")
    import shutil
    shutil.rmtree(carpeta_temp)
    print(f"✓ Carpeta temporal eliminada")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETADO: {nombre_base}")
    print(f"{'='*70}\n")


def main():
    """
    Función principal que procesa ambos datasets con CNN.
    """
    # Obtener la ruta del script para construir rutas absolutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proyecto_root = os.path.join(script_dir, '..', '..')
    
    # Rutas a los datasets preprocesados
    datasets = [
        (os.path.join(proyecto_root, "dataset", "dataset_zips_procesados", "Cats_Dogs_Foxes_procesado.zip"), "Cats_Dogs_Foxes"),
        (os.path.join(proyecto_root, "dataset", "dataset_zips_procesados", "Vocals_UPIIT2025_procesado.zip"), "Vocals_UPIIT2025")
    ]
    
    print("\n" + "="*70)
    print("EXTRACCIÓN DE CARACTERÍSTICAS CNN (VGG16 con TensorFlow)")
    print("="*70)
    
    # Procesar cada dataset
    for ruta_zip, nombre_base in datasets:
        if os.path.exists(ruta_zip):
            procesar_dataset_completo(ruta_zip, nombre_base)
        else:
            print(f"⚠ ADVERTENCIA: No se encontró {ruta_zip}")
    
    print("\n" + "="*70)
    print("✓ PROCESO COMPLETO FINALIZADO")
    print("="*70)
    

if __name__ == "__main__":
    main()
