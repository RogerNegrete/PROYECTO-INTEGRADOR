import os
import shutil
import zipfile
import re

from Descarga_Data import descargar_dataset_kaggle
from procesamiento_imagenes import procesar_dataset


def comprimir_dataset(ruta_carpeta, ruta_zip):
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for raiz, _, archivos in os.walk(ruta_carpeta):
            for archivo in archivos:
                ruta = os.path.join(raiz, archivo)
                zipf.write(ruta, os.path.relpath(ruta, ruta_carpeta))


def eliminar_carpeta(ruta):
    if os.path.exists(ruta):
        shutil.rmtree(ruta)


def descomprimir_zip(ruta_zip, carpeta_salida):
    with zipfile.ZipFile(ruta_zip, "r") as zipf:
        zipf.extractall(carpeta_salida)


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



def copiar_dataset_completo(ruta_origen, ruta_destino):
    shutil.copytree(ruta_origen, ruta_destino)


def flujo_dataset(dataset_id, nombre_base):

    carpeta_zips_originales = "dataset_zips_originales"
    carpeta_zips_procesados = "dataset_zips_procesados"
    carpeta_tmp = "tmp"
    carpeta_proc = "procesado"
    carpeta_filtrada = "filtrado"

    os.makedirs(carpeta_zips_originales, exist_ok=True)
    os.makedirs(carpeta_zips_procesados, exist_ok=True)
    os.makedirs(carpeta_tmp, exist_ok=True)
    os.makedirs(carpeta_proc, exist_ok=True)
    os.makedirs(carpeta_filtrada, exist_ok=True)

    # 1. Descargar dataset
    descargar_dataset_kaggle(dataset_id, carpeta_zips_originales)

    zip_original = next(
        f for f in os.listdir(carpeta_zips_originales) if f.endswith(".zip")
    )
    ruta_zip_original = os.path.join(carpeta_zips_originales, zip_original)

    # 2. Descomprimir
    ruta_tmp = os.path.join(carpeta_tmp, nombre_base)
    descomprimir_zip(ruta_zip_original, ruta_tmp)

    # 3. Corregir nombres de archivos
    corregir_nombres_archivos(ruta_tmp)

    # 4. Copiar TODO el dataset (sin condicional)
    ruta_filtrada = os.path.join(carpeta_filtrada, nombre_base)
    print("\nUsando todas las imágenes del dataset...")
    copiar_dataset_completo(ruta_tmp, ruta_filtrada)

    # 5. Procesar imágenes (resize incluido)
    ruta_proc = os.path.join(carpeta_proc, nombre_base)
    procesar_dataset(ruta_filtrada, ruta_proc)

    # 5.5. Resumen del dataset procesado
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

    # 6. Comprimir dataset procesado
    ruta_zip_final = os.path.join(
        carpeta_zips_procesados, f"{nombre_base}_procesado.zip"
    )
    comprimir_dataset(ruta_proc, ruta_zip_final)

    # 7. Limpieza
    os.remove(ruta_zip_original)
    eliminar_carpeta(ruta_tmp)
    eliminar_carpeta(ruta_filtrada)
    eliminar_carpeta(ruta_proc)

    print("\nFlujo completado correctamente.")
