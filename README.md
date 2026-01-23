# PROYECTO INTEGRADOR

INTEGRANTES:
- Alexander Calapaqui
- Manuel Coyago
- Anthony Herrera
- Roger Negrete


## Funcionalidades

- **Descarga de Datasets:** Descarga automática de datasets desde Kaggle utilizando la API oficial.
- **Procesamiento de Imágenes:** Aplica técnicas de preprocesamiento como mejora de contraste (CLAHE), eliminación de ruido (filtro Gaussiano), umbralización (Otsu) y rescalado uniforme a 224x224 píxeles.
- **Corrección Automática de Nombres:** Detecta y corrige errores tipográficos en nombres de archivos de imágenes, como sufijos extraños después de las extensiones (.jpg, .png, .jpeg).
- **Compresión de Datasets:** Comprime los datasets procesados en archivos ZIP para optimizar el almacenamiento.
- **Resumen de Procesamiento:** Muestra por consola un resumen detallado de cada dataset procesado, incluyendo el total de imágenes y el conteo por clase.
- **Limpieza Automática:** Elimina archivos temporales y carpetas intermedias al finalizar el proceso.

## Requisitos

- Python 3.7+
- Librerías: `kaggle`, `opencv-python`, `numpy`
- API Key de Kaggle configurada (ver instalación)

## Instalación

1. Clona el repositorio:
   ```
   git clone <url-del-repositorio>
   cd PROYECTO-INTEGRADOR
   ```

2. Instala las dependencias:
   ```
   pip install kaggle opencv-python numpy
   ```

3. Configura la API de Kaggle:
   - Ve a [Kaggle Account](https://www.kaggle.com/account) y descarga tu `kaggle.json`.
   - Coloca el archivo en `~/.kaggle/kaggle.json` (Linux/Mac) o `C:\Users\<tu_usuario>\.kaggle\kaggle.json` (Windows).
   - Asegúrate de que tenga permisos adecuados.

## Uso

Ejecuta el script principal para procesar los datasets definidos:

```
python main.py
```

Esto descargará, procesará y comprimirá los datasets especificados en `main.py` (actualmente: "csareduardoarenas/vocals-upiit2025" y "snmahsa/animal-image-dataset-cats-dogs-and-foxes").

### Salida por Consola

Al finalizar, verás un resumen como:

```
--- Resumen del dataset 'Vocals_UPIIT2025' ---
Total imágenes procesadas: 500
Clase 'A': 100 imágenes
Clase 'E': 100 imágenes
...
```

## Optimizaciones Realizadas

- **Descarga Eficiente:** Los datasets se descargan como ZIP y se descomprimen solo cuando es necesario, minimizando el uso de espacio.
- **Procesamiento en Lote:** Todas las imágenes se procesan de manera secuencial, aplicando rescalado uniforme para consistencia.
- **Corrección Automática:** La función `corregir_nombres_archivos` usa expresiones regulares para detectar y corregir sufijos extraños en nombres de archivos, evitando errores manuales.
- **Compresión Selectiva:** Solo los archivos procesados se comprimen, reduciendo el tamaño final.
- **Limpieza Integral:** Eliminación automática de archivos temporales para mantener el directorio limpio.

## Estructura del Proyecto

- `main.py`: Script principal que ejecuta el flujo para múltiples datasets.
- `data_img.py`: Módulo principal del flujo de datos, incluyendo descarga, procesamiento y compresión.
- `Descarga_Data.py`: Funciones para la descarga desde Kaggle.
- `procesamiento_imagenes.py`: Funciones de preprocesamiento de imágenes.
- `README.md`: Este archivo.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request en el repositorio.

## Licencia

Este proyecto está bajo la Licencia MIT.
