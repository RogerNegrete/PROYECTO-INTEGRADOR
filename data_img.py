import os
import shutil
import zipfile
import random

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


def contar_instancias_por_clase(ruta_dataset):
    print("\n Número total de instancias por clase:\n")
    conteo = {}

    for clase in os.listdir(ruta_dataset):
        ruta_clase = os.path.join(ruta_dataset, clase)
        if not os.path.isdir(ruta_clase):
            continue

        imagenes = [
            f for f in os.listdir(ruta_clase)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        conteo[clase] = len(imagenes)
        print(f"Clase '{clase}': {len(imagenes)} imágenes")

    return conteo


def seleccionar_100_por_clase(ruta_dataset, ruta_salida, max_imgs=100):
    os.makedirs(ruta_salida, exist_ok=True)

    for clase in os.listdir(ruta_dataset):
        ruta_clase = os.path.join(ruta_dataset, clase)
        if not os.path.isdir(ruta_clase):
            continue

        imagenes = [
            f for f in os.listdir(ruta_clase)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        seleccion = random.sample(imagenes, min(max_imgs, len(imagenes)))

        ruta_dest_clase = os.path.join(ruta_salida, clase)
        os.makedirs(ruta_dest_clase, exist_ok=True)

        for img in seleccion:
            shutil.copy(
                os.path.join(ruta_clase, img),
                os.path.join(ruta_dest_clase, img)
            )


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

    # 3. Contar instancias por clase
    contar_instancias_por_clase(ruta_tmp)

    # 4. Preguntar si desea filtrar a 100 imágenes
    opcion = input(
        "\n¿Deseas seleccionar solo 100 imágenes por clase? (s/n): "
    ).strip().lower()

    ruta_filtrada = os.path.join(carpeta_filtrada, nombre_base)

    if opcion == "s":
        print("\n Seleccionando 100 imágenes por clase...")
        seleccionar_100_por_clase(ruta_tmp, ruta_filtrada)
    else:
        print("\n Usando todas las imágenes del dataset...")
        copiar_dataset_completo(ruta_tmp, ruta_filtrada)

    # 5. Procesar imágenes
    ruta_proc = os.path.join(carpeta_proc, nombre_base)
    procesar_dataset(ruta_filtrada, ruta_proc)

    # 6. Comprimir dataset procesado
    ruta_zip_final = os.path.join(
        carpeta_zips_procesados, f"{nombre_base}_procesado.zip"
    )
    comprimir_dataset(ruta_proc, ruta_zip_final)

    # 7. Limpieza total
    os.remove(ruta_zip_original)
    eliminar_carpeta(ruta_tmp)
    eliminar_carpeta(ruta_filtrada)
    eliminar_carpeta(ruta_proc)

    print("\n Flujo completado correctamente.")
