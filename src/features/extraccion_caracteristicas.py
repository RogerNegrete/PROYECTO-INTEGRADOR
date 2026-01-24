import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from math import copysign, log10


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
        tuple: (imagenes, etiquetas, nombres_clases)
            - imagenes: Lista de imágenes en formato numpy array
            - etiquetas: Lista de nombres de clases (strings)
            - nombres_clases: Lista de nombres únicos de clases
    """
    imagenes = []
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
            img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                imagenes.append(img)
                etiquetas.append(clase)  # Guardar nombre de clase en lugar de número
    
    print(f"\n✓ Total imágenes cargadas: {len(imagenes)}")
    
    if len(imagenes) == 0:
        raise ValueError("No se cargaron imágenes. Verifica la estructura del dataset.")
    
    return np.array(imagenes), np.array(etiquetas), clases


def extraer_momentos_hu(img):
    """
    Extrae 7 momentos de Hu de una imagen.
    Aplica escala logarítmica para normalizar los valores.
    
    Args:
        img: Imagen 2D en escala de grises
        
    Returns:
        array: 7 momentos de Hu en escala logarítmica
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    moments = cv2.moments(img)
    huMoments = cv2.HuMoments(moments).flatten()
    
    # Convertir a escala logarítmica
    for i in range(7):
        if huMoments[i] != 0:
            huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
        else:
            huMoments[i] = 0
    
    return huMoments


def extraer_sift(img, n_keypoints=100):
    """
    Extrae descriptores SIFT de una imagen.
    
    Args:
        img: Imagen 2D en escala de grises
        n_keypoints: Número máximo de keypoints a retener
        
    Returns:
        array: Descriptores SIFT aplanados (vector de tamaño fijo)
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    # Si no hay suficientes keypoints, rellenar con ceros
    if descriptors is None:
        descriptors = np.zeros((n_keypoints, 128))
    elif len(descriptors) < n_keypoints:
        padding = np.zeros((n_keypoints - len(descriptors), 128))
        descriptors = np.vstack([descriptors, padding])
    elif len(descriptors) > n_keypoints:
        descriptors = descriptors[:n_keypoints]
    
    return descriptors.flatten()


def extraer_hog(img):
    """
    Extrae características HOG (Histogram of Oriented Gradients) de una imagen.
    
    Args:
        img: Imagen 2D en escala de grises
        
    Returns:
        array: Descriptor HOG
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Parámetros HOG
    win_size = (img.shape[1] // 16 * 16, img.shape[0] // 16 * 16)  # Debe ser múltiplo de 16
    img_resized = cv2.resize(img, win_size)
    
    hog = cv2.HOGDescriptor(
        win_size,
        (16, 16),  # block size
        (8, 8),    # block stride
        (8, 8),    # cell size
        9          # nbins
    )
    
    descriptor = hog.compute(img_resized)
    return descriptor.flatten()


def extraer_caracteristicas_dataset(imagenes, etiquetas, metodo='hu'):
    """
    Extrae características de todas las imágenes del dataset.
    
    Args:
        imagenes: Array de imágenes
        etiquetas: Array de etiquetas
        metodo: 'hu', 'sift' o 'hog'
        
    Returns:
        tuple: (caracteristicas, etiquetas)
    """
    print(f"\n{'='*60}")
    print(f"Extrayendo características usando método: {metodo.upper()}")
    print(f"{'='*60}")
    
    caracteristicas = []
    
    for i, img in enumerate(imagenes):
        if (i + 1) % 100 == 0:
            print(f"Procesadas {i + 1}/{len(imagenes)} imágenes...")
        
        if metodo == 'hu':
            features = extraer_momentos_hu(img)
        elif metodo == 'sift':
            features = extraer_sift(img)
        elif metodo == 'hog':
            features = extraer_hog(img)
        else:
            raise ValueError(f"Método '{metodo}' no reconocido. Use 'hu', 'sift' o 'hog'")
        
        caracteristicas.append(features)
    
    print(f"✓ Completado: {len(caracteristicas)} vectores de características")
    return np.array(caracteristicas), etiquetas


def guardar_caracteristicas(caracteristicas, etiquetas, nombres_clases, nombre_base, metodo):
    """
    Guarda las características en formato CSV.
    
    Args:
        caracteristicas: Array de características
        etiquetas: Array de etiquetas (nombres de clases)
        nombres_clases: Lista de nombres únicos de clases
        nombre_base: Nombre base del archivo
        metodo: Método usado ('hu', 'sift', 'hog')
    """
    if len(caracteristicas) == 0:
        print(f"⚠ No hay características para guardar ({metodo})")
        return
    
    # Guardar en el mismo directorio donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archivo_csv = os.path.join(script_dir, f"{nombre_base}_{metodo}.csv")
    
    # Guardar como CSV
    print(f"\nGuardando características en CSV: {archivo_csv}")
    n_features = caracteristicas.shape[1]
    columnas = [f"feature_{i}" for i in range(n_features)]
    
    df = pd.DataFrame(caracteristicas, columns=columnas)
    df["clase"] = etiquetas
    df.to_csv(archivo_csv, index=False)
    print(f"✓ Guardado CSV: {archivo_csv}")


def procesar_dataset_completo(ruta_zip, nombre_base, metodos=['hu', 'sift', 'hog']):
    """
    Pipeline completo: descomprime, carga imágenes y extrae características.
    
    Args:
        ruta_zip: Ruta al archivo ZIP con imágenes preprocesadas
        nombre_base: Nombre base para los archivos de salida
        metodos: Lista de métodos a aplicar ('hu', 'sift', 'hog')
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO DATASET: {nombre_base}")
    print(f"{'='*70}")
    
    # 1. Descomprimir
    carpeta_temp = f"temp_{nombre_base}"
    descomprimir_zip(ruta_zip, carpeta_temp)
    
    # 2. Cargar imágenes
    imagenes, etiquetas, nombres_clases = cargar_imagenes_desde_carpetas(carpeta_temp)
    
    # 3. Extraer características con cada método
    for metodo in metodos:
        caracteristicas, _ = extraer_caracteristicas_dataset(imagenes, etiquetas, metodo)
        guardar_caracteristicas(caracteristicas, etiquetas, nombres_clases, nombre_base, metodo)
    
    # 4. Limpiar carpeta temporal
    print(f"\nLimpiando carpeta temporal: {carpeta_temp}")
    import shutil
    shutil.rmtree(carpeta_temp)
    print(f"✓ Carpeta temporal eliminada")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETADO: {nombre_base}")
    print(f"{'='*70}\n")


def main():
    """
    Función principal que procesa ambos datasets.
    """
    # Obtener la ruta del script para construir rutas absolutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proyecto_root = os.path.join(script_dir, '..', '..')
    
    # Rutas a los datasets preprocesados
    datasets = [
        (os.path.join(proyecto_root, "dataset/dataset_zips_procesados/Vocals_UPIIT2025_procesado.zip"), "Vocals_UPIIT2025"),
        (os.path.join(proyecto_root, "dataset/dataset_zips_procesados/Cats_Dogs_Foxes_procesado.zip"), "Cats_Dogs_Foxes")
    ]
    
    # Métodos de extracción de características
    metodos = ['hu', 'sift', 'hog']
    
    print("\n" + "="*70)
    print("EXTRACCIÓN DE CARACTERÍSTICAS DE DATASETS PREPROCESADOS")
    print("="*70)
    print(f"Métodos a aplicar: {', '.join([m.upper() for m in metodos])}")
    print("="*70)
    
    # Procesar cada dataset
    for ruta_zip, nombre_base in datasets:
        if os.path.exists(ruta_zip):
            procesar_dataset_completo(ruta_zip, nombre_base, metodos)
        else:
            print(f"⚠ ADVERTENCIA: No se encontró {ruta_zip}")
    
    print("\n" + "="*70)
    print("✓ PROCESO COMPLETO FINALIZADO")
    print("="*70)
    

if __name__ == "__main__":
    main()
