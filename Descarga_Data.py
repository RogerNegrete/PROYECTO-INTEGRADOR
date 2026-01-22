import os
from kaggle.api.kaggle_api_extended import KaggleApi


def descargar_dataset_kaggle(dataset_id, ruta_destino):
    """
    Descarga un dataset desde Kaggle y conserva el ZIP original
    """

    api = KaggleApi()
    api.authenticate()
    print("API Kaggle autenticada correctamente")

    os.makedirs(ruta_destino, exist_ok=True)


    api.dataset_download_files(
        dataset_id,
        path=ruta_destino,
        unzip=False
    )

    print(f"ZIP original descargado en: {ruta_destino}")
