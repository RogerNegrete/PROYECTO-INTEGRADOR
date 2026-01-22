import os
import cv2
import numpy as np


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

                # Guardar imagen
                ruta_guardado = os.path.join(carpeta_salida, archivo)
                cv2.imwrite(ruta_guardado, umbral)

        print(f"Procesada carpeta: {raiz}")
