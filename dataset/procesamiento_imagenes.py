import os
import shutil
import zipfile
import re
import cv2
import kagglehub


def comprimir_dataset(ruta_carpeta, ruta_zip):
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for raiz, _, archivos in os.walk(ruta_carpeta):
            for archivo in archivos:
                ruta = os.path.join(raiz, archivo)
                zipf.write(ruta, os.path.relpath(ruta, ruta_carpeta))


def eliminar_carpeta(ruta):
    if os.path.exists(ruta):
        shutil.rmtree(ruta)


def corregir_nombres_archivos(ruta_dataset):
    """
    Corrige nombres de archivos malformados en el dataset, detectando y removiendo sufijos extraños después de extensiones válidas.
    """
    extensiones_validas = ['.jpg', '.jpeg', '.png']
    patron = re.compile(r'(.+)(' + '|'.join(extensiones_validas) + r')(.*)$', re.IGNORECASE)
    
    for raiz, _, archivos in os.walk(ruta_dataset):
        for archivo in archivos:
            match = patron.match(archivo)
            if match and match.group(3):
                # Tiene sufijo extra después de la extensión
                nombre_base = match.group(1)
                extension = match.group(2)
                sufijo_extra = match.group(3)
                nuevo_nombre = nombre_base + extension
                ruta_vieja = os.path.join(raiz, archivo)
                ruta_nueva = os.path.join(raiz, nuevo_nombre)
                os.rename(ruta_vieja, ruta_nueva)
                print(f"Corregido automáticamente: {ruta_vieja} -> {ruta_nueva} (removido sufijo '{sufijo_extra}')")


def mejorar_contraste(imagen):
    """
    Mejora el contraste usando CLAHE
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contraste = clahe.apply(gris)

    return contraste


def eliminar_ruido(imagen):
    """
    Elimina ruido usando filtro Gaussiano
    """
    ruido = cv2.GaussianBlur(imagen, (5, 5), 0)
    return ruido


def umbralizar_imagen(imagen):
    """
    Aplica umbralización de Otsu
    """
    _, umbral = cv2.threshold(
        imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return umbral


def redimensionar_imagen(imagen, tamaño=(224, 224)):
    """
    Redimensiona la imagen al tamaño especificado
    """
    return cv2.resize(imagen, tamaño)


def procesar_dataset(ruta_entrada, ruta_salida):
    """
    Procesa todas las imágenes de un dataset
    """

    os.makedirs(ruta_salida, exist_ok=True)

    for raiz, _, archivos in os.walk(ruta_entrada):
        for archivo in archivos:
            if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
                
                ruta_imagen = os.path.join(raiz, archivo)

                # Mantener estructura de carpetas
                ruta_relativa = os.path.relpath(raiz, ruta_entrada)
                carpeta_salida = os.path.join(ruta_salida, ruta_relativa)
                os.makedirs(carpeta_salida, exist_ok=True)

                # Leer imagen
                imagen = cv2.imread(ruta_imagen)

                if imagen is None:
                    continue

                # Procesamiento
                contraste = mejorar_contraste(imagen)
                sin_ruido = eliminar_ruido(contraste)
                umbral = umbralizar_imagen(sin_ruido)
                redimensionada = redimensionar_imagen(umbral)

                # Guardar imagen
                ruta_guardado = os.path.join(carpeta_salida, archivo)
                cv2.imwrite(ruta_guardado, redimensionada)

        print(f"Procesada carpeta: {raiz}")


def flujo_dataset(dataset_id, nombre_base):
    carpeta_base = "dataset"
    carpeta_zips_procesados = os.path.join(carpeta_base, "dataset_zips_procesados")
    carpeta_proc = os.path.join(carpeta_base, "procesado")

    os.makedirs(carpeta_zips_procesados, exist_ok=True)
    os.makedirs(carpeta_proc, exist_ok=True)

    # 1. Descargar dataset
    print(f"Descargando dataset {nombre_base} desde Kaggle...")
    ruta_tmp = kagglehub.dataset_download(dataset_id)
    print(f"✓ Dataset descargado en: {ruta_tmp}")

    # 2. Corregir nombres de archivos
    corregir_nombres_archivos(ruta_tmp)

    # 3. Procesar imágenes directamente
    ruta_proc = os.path.join(carpeta_proc, nombre_base)
    procesar_dataset(ruta_tmp, ruta_proc)

    # 4. Resumen del dataset procesado
    print(f"\n--- Resumen del dataset '{nombre_base}' ---")
    conteo_procesado = {}
    total_procesadas = 0
    for raiz, _, archivos in os.walk(ruta_proc):
        for archivo in archivos:
            if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
                clase = os.path.basename(raiz)
                if clase not in conteo_procesado:
                    conteo_procesado[clase] = 0
                conteo_procesado[clase] += 1
                total_procesadas += 1
    print(f"Total imágenes procesadas: {total_procesadas}")
    for clase, num in conteo_procesado.items():
        print(f"Clase '{clase}': {num} imágenes")

    # 5. Comprimir dataset procesado
    ruta_zip_final = os.path.join(carpeta_zips_procesados, f"{nombre_base}_procesado.zip")
    comprimir_dataset(ruta_proc, ruta_zip_final)

    # 6. Limpieza
    eliminar_carpeta(ruta_proc)
    if os.path.exists(carpeta_proc) and not os.listdir(carpeta_proc):
        os.rmdir(carpeta_proc)

    print(f"✓ Dataset guardado en: {ruta_zip_final}\n")


def main():
    flujo_dataset("csareduardoarenas/vocals-upiit2025", "Vocals_UPIIT2025")
    flujo_dataset("snmahsa/animal-image-dataset-cats-dogs-and-foxes", "Cats_Dogs_Foxes")


if __name__ == "__main__":
    main()
