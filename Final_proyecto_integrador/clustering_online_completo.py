"""
Sistema de Clustering Online PURO
==================================
Sistema de clustering online que procesa imágenes UNA POR UNA en tiempo real.
Cada imagen entra al clustering INMEDIATAMENTE después de extraer características.

CARACTERÍSTICAS:
- Sin PCA (eliminado para evitar dependencias globales)
- Sin esperar batches (procesamiento imagen por imagen)
- Sin reprocesar el pasado (verdaderamente incremental)
- Normalización L2 por punto (independiente)

IMPORTANTE: TODO SE MANTIENE EN MEMORIA - NO SE GUARDA EN DISCO
"""

import os
import cv2
import numpy as np
import io
import tensorflow as tf
import threading
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import (
    silhouette_score, 
    adjusted_rand_score, 
    normalized_mutual_info_score, 
    adjusted_mutual_info_score
)
from scipy.spatial.distance import cdist
from pathlib import Path
from math import copysign, log10
from PIL import Image

class OnlineKMeansCapacitado:
    """
    K-Means Online CORREGIDO.
    Recibe correctamente k, capacities y threshold.
    """
    def __init__(self, k, capacities, threshold=0.65):
        self.k = k
        self.capacities = capacities[:]
        self.remaining = capacities[:]
        self.threshold = threshold  # <--- AQUI SE GUARDA EL PARAMETRO CONFIGURABLE
        self.centers = []
        self.counts = []
        self.cluster_centers_ = None 
        self.labels_ = [] 
    
    def partial_fit(self, x):
        x = np.asarray(x).astype(float)
        
        # FASE 1: ARRANQUE (Usando lógica de Líderes)
        if len(self.centers) < self.k:
            if len(self.centers) == 0:
                self._create_new_cluster(x)
                return 0
            
            distances = [np.linalg.norm(x - c) for c in self.centers]
            min_dist = np.min(distances)
            nearest_idx = np.argmin(distances)
            
            # AQUI SE USA EL UMBRAL DINAMICO (NO QUEMADO)
            if min_dist > self.threshold:
                self._create_new_cluster(x)
                return len(self.centers) - 1
            else:
                # Es similar, intentar unir
                if self.remaining[nearest_idx] > 0:
                    self._update_cluster(nearest_idx, x)
                    return nearest_idx
                else:
                    self._create_new_cluster(x)
                    return len(self.centers) - 1

        # FASE 2: ASIGNACIÓN (Cuando ya hay K centros - Lógica Fallback)
        distances = []
        for i, c in enumerate(self.centers):
            if self.remaining[i] > 0:
                distances.append(np.linalg.norm(x - c))
            else:
                distances.append(np.inf)
        
        # Forzar asignación al más cercano disponible
        if np.all(np.isinf(distances)):
             dists_real = [np.linalg.norm(x - c) for c in self.centers]
             idx = np.argmin(dists_real)
        else:
             idx = int(np.argmin(distances))
        
        self._update_cluster(idx, x)
        return idx

    def _create_new_cluster(self, x):
        self.centers.append(x.copy())
        self.counts.append(1)
        idx = len(self.centers) - 1
        self.remaining[idx] -= 1
        self.labels_.append(idx)
        self.cluster_centers_ = np.array(self.centers)

    def _update_cluster(self, idx, x):
        self.counts[idx] += 1
        self.remaining[idx] -= 1
        eta = 1.0 / self.counts[idx]
        self.centers[idx] = (1.0 - eta) * self.centers[idx] + eta * x
        self.labels_.append(idx)
        self.cluster_centers_ = np.array(self.centers)

    def get_distribution(self):
        return [cap - rem for cap, rem in zip(self.capacities, self.remaining)]
    
    def reset(self):
        self.remaining = self.capacities[:]
        self.centers = []
        self.counts = []
        self.cluster_centers_ = None
        self.labels_ = []

class OnlineDynamicClustering:
    """
    Clustering Online Dinámico basado en reglas de líderes y capacidad.
    Implementación estricta del documento "Explicación del Algoritmo de Clustering Online".
    """

    def __init__(self, max_k, capacities, threshold=0.85):
        self.max_k = max_k
        self.capacities = capacities[:]
        self.remaining = capacities[:]
        self.threshold = threshold  # Umbral fijo 0.85
        self.centers = []
        self.counts = []
        self.labels_ = []
        self.cluster_centers_ = None

    def partial_fit(self, x):
        """
        Procesa una imagen siguiendo el ciclo de vida descrito en el PDF.
        """
        x = np.asarray(x).astype(float)
        
        # ---------------------------------------------------------
        # 1. ARRANQUE (Caso Inicial) 
        # "Si no existen grupos todavía, la primera imagen funda automáticamente el Grupo 1."
        # ---------------------------------------------------------
        if len(self.centers) == 0:
            return self._create_new_cluster(x)

        # ---------------------------------------------------------
        # 2. COMPARACIÓN 
        # "Se mide la distancia euclidiana entre la nueva imagen y los centros..."
        # ---------------------------------------------------------
        distances = [np.linalg.norm(x - c) for c in self.centers]
        sorted_indices = np.argsort(distances) # Ordenar candidatos del más cercano al más lejano
        
        nearest_idx = sorted_indices[0]
        nearest_dist = distances[nearest_idx]
        
        assigned_cluster = -1

        # ---------------------------------------------------------
        # 3. DECISIÓN PRINCIPAL 
        # ---------------------------------------------------------
        
        # CASO A: SIMILAR (Cerca) (d <= 0.85)
        if nearest_dist <= self.threshold:
            # Intentar unirse al grupo. Si está lleno, buscar siguiente opción válida.
            for idx in sorted_indices:
                # Verificamos que el candidato alternativo también sea similar (<= 0.85)
                if distances[idx] <= self.threshold:
                    if self.remaining[idx] > 0: # Si hay espacio
                        self._update_cluster(idx, x)
                        assigned_cluster = idx
                        break
                    # Si está LLENO -> Se busca el siguiente
                else:
                    # Si el siguiente candidato ya está lejos (>0.85), paramos.
                    break
        
        # CASO B: DIFERENTE (Lejos) o SIN CUPO EN SIMILARES 
        if assigned_cluster == -1:
            # "La imagen no encaja con lo conocido"
            
            # Si aún se permiten más grupos -> Crea un Nuevo Grupo
            if len(self.centers) < self.max_k:
                return self._create_new_cluster(x)
            
            # ---------------------------------------------------------
            # 4. EMERGENCIA (Fallback) 
            # ---------------------------------------------------------
            # "Se ejecuta únicamente si no fue posible asignar... ni crear un nuevo grupo"
            # Se fuerza asignación al más cercano ignorando el umbral
            return self._force_assign_nearest_available(x, sorted_indices)

        # Guardar historial para métricas
        self.labels_.append(assigned_cluster)
        self.cluster_centers_ = np.array(self.centers)
        return assigned_cluster

    def _create_new_cluster(self, x):
        """Crea un nuevo grupo (fundador)"""
        cluster_idx = len(self.centers)
        
        # Manejo dinámico de capacidades si crecen los clusters
        if cluster_idx >= len(self.capacities):
            cap = self.capacities[-1] if self.capacities else 5
            self.capacities.append(cap)
            self.remaining.append(cap)
            
        self.centers.append(x.copy())
        self.counts.append(1)
        self.remaining[cluster_idx] -= 1
        
        self.labels_.append(cluster_idx)
        self.cluster_centers_ = np.array(self.centers)
        return cluster_idx

    def _update_cluster(self, idx, x):
        """
        Actualización (Aprendizaje del Centroide)
        Formula de Media Móvil: Centro_nuevo = (1-n)Centro_viejo + n*Imagen_nueva
        """
        self.counts[idx] += 1
        self.remaining[idx] -= 1
        
        # n es el factor de peso (eta)
        eta = 1.0 / self.counts[idx]
        self.centers[idx] = (1.0 - eta) * self.centers[idx] + eta * x
        
        self.labels_.append(idx)
        self.cluster_centers_ = np.array(self.centers)

    def _force_assign_nearest_available(self, x, sorted_indices):
        """Fuerza la asignación al cluster disponible más cercano"""
        for idx in sorted_indices:
            if self.remaining[idx] > 0:
                self._update_cluster(idx, x)
                return idx
        
        # Si absolutamente todo está lleno (caso extremo no contemplado pero posible)
        # forzamos al más cercano aunque desborde capacidad (para no perder el dato)
        idx = sorted_indices[0]
        self._update_cluster(idx, x)
        return idx
        
    def get_distribution(self):
        return [cap - rem for cap, rem in zip(self.capacities, self.remaining)]


def dunn_index(X, labels):
    """
    Calcula el índice de Dunn para evaluar la calidad del clustering.
    Dunn = min_inter_cluster_distance / max_intra_cluster_distance
    
    EXACTAMENTE como el ejemplo proporcionado.
    """
    clusters = np.unique(labels)
    
    if len(clusters) < 2:
        return 0.0
    
    # distancia mínima entre clusters
    inter_cluster_dist = np.inf
    for i in clusters:
        for j in clusters:
            if i >= j:
                continue
            points_i = X[labels == i]
            points_j = X[labels == j]
            if len(points_i) > 0 and len(points_j) > 0:
                dist = np.min([
                    np.linalg.norm(x - y)
                    for x in points_i
                    for y in points_j
                ])
                inter_cluster_dist = min(inter_cluster_dist, dist)
    
    # diámetro máximo intra-cluster
    intra_cluster_dist = 0
    for i in clusters:
        points = X[labels == i]
        if len(points) > 1:
            dist = np.max([
                np.linalg.norm(x - y)
                for x in points
                for y in points
            ])
            intra_cluster_dist = max(intra_cluster_dist, dist)
    
    if intra_cluster_dist == 0:
        return 0.0
    
    return inter_cluster_dist / intra_cluster_dist


class ClusteringOnlineSystem:
    """
    Sistema de clustering online incremental con restricciones de tamaño.
    """
    
    def __init__(self, metodo='hog', n_clusters=3, cluster_sizes=None, storage_dir='storage', distance_threshold=0.75):
        """
        Inicializa el sistema de clustering online.
        
        Args:
            metodo: 'cnn', 'hog', 'sift', 'hu_moments'
            n_clusters: Número de clusters
            cluster_sizes: Lista de tamaños exactos por cluster (None = sin restricción)
                          Ejemplo: [5, 7, 8] para 3 clusters con esos tamaños
            storage_dir: NO SE USA - todo se mantiene en memoria
        """
        self.metodo = metodo.lower()
        self.n_clusters = n_clusters
        self.cluster_sizes = cluster_sizes
        self.distance_threshold = distance_threshold
        # storage_dir se mantiene por compatibilidad pero NO se usa
        self.storage_dir = Path(storage_dir) if storage_dir else Path('storage')

        # Lock para proteger cambios concurrentes
        self.lock = threading.Lock()
        
        # Modelo de clustering online (SIN PCA, SIN SCALER GLOBAL)
        self.model = None
        # NO usamos StandardScaler - usamos normalización L2 por punto
        # NO usamos PCA - procesamiento directo
        
        # Datos acumulados EN MEMORIA (no se guardan en disco)
        self.X_accumulated = []  # Características acumuladas
        self.labels_accumulated = []  # Etiquetas ground truth (si existen)
        self.filenames_accumulated = []  # Nombres de archivos
        self.predictions_accumulated = []  # Predicciones del modelo
        
        # Extractor de características
        self.feature_extractor = None
        if self.metodo == 'cnn':
            print("Cargando VGG16 para extracción CNN...")
            self.feature_extractor = VGG16(weights='imagenet', include_top=False, 
                                          input_shape=(224, 224, 3))
            print("✓ VGG16 cargado")
        
        # Flag para saber si ya tenemos datos
        self.has_data = False
        self._history_ingested = False
        
        print("✓ Sistema inicializado - DATOS EN MEMORIA (no se guarda en disco)")
    
    # ==================== PREPROCESAMIENTO ====================
    
    def preprocesar_imagen(self, imagen_input):
        """
        Preprocesa una imagen: mejora contraste, elimina ruido y UMBRALIZA (Blanco/Negro).
        """
        # Caso 1: bytes (desde memoria)
        if isinstance(imagen_input, bytes):
            try:
                pil_image = Image.open(io.BytesIO(imagen_input))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                imagen = np.array(pil_image)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"No se pudo cargar la imagen desde bytes: {str(e)}")
        
        # Caso 2: ruta a archivo
        elif isinstance(imagen_input, (str, Path)):
            ruta = str(imagen_input)
            # (Validaciones de extensión omitidas por brevedad, usar lógica estándar)
            imagen = cv2.imread(ruta)
            if imagen is None:
                raise ValueError(f"No se pudo cargar la imagen: {ruta}")
        
        # Caso 3: numpy array
        else:
            imagen = imagen_input
        
        # 1. Escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # 2. Mejora de contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contraste = clahe.apply(gris)
        
        # 3. Eliminar ruido
        sin_ruido = cv2.GaussianBlur(contraste, (5, 5), 0)
        
        # 4. UMBRALIZACIÓN (ESTA ES LA CLAVE QUE HABÍAMOS QUITADO)
        # Convierte la imagen a puro Blanco y Negro. Hu Moments necesita esto.
        _, umbral = cv2.threshold(sin_ruido, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Redimensionar
        redimensionada = cv2.resize(umbral, (224, 224))
        
        return redimensionada
    
    # ==================== EXTRACCIÓN DE CARACTERÍSTICAS ====================
    
    def extraer_caracteristicas_cnn(self, imagen):
        """Extrae características CNN usando VGG16"""
        if len(imagen.shape) == 2:
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        img_rgb = cv2.resize(img_rgb, (224, 224))
        img_rgb = preprocess_input(img_rgb)
        img_rgb = np.expand_dims(img_rgb, axis=0)
        
        features = self.feature_extractor.predict(img_rgb, verbose=0)
        return features.flatten()
    
    def extraer_caracteristicas_hu_moments(self, imagen):
        """Extrae 7 momentos de Hu con escala logarítmica"""
        if imagen.max() <= 1.0:
            imagen = (imagen * 255).astype(np.uint8)
        else:
            imagen = imagen.astype(np.uint8)
        
        # Calculamos momentos directamente (la imagen ya viene en B/N)
        moments = cv2.moments(imagen)
        huMoments = cv2.HuMoments(moments).flatten()
        
        # Escala logarítmica
        for i in range(7):
            if huMoments[i] != 0:
                huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
            else:
                huMoments[i] = 0
        
        return huMoments
    
    def extraer_caracteristicas_sift(self, imagen, n_keypoints=100):
        """Extrae descriptores SIFT"""
        if imagen.max() <= 1.0:
            imagen = (imagen * 255).astype(np.uint8)
        else:
            imagen = imagen.astype(np.uint8)
        
        sift = cv2.SIFT_create(nfeatures=n_keypoints)
        keypoints, descriptors = sift.detectAndCompute(imagen, None)
        
        if descriptors is None:
            descriptors = np.zeros((n_keypoints, 128))
        elif len(descriptors) < n_keypoints:
            padding = np.zeros((n_keypoints - len(descriptors), 128))
            descriptors = np.vstack([descriptors, padding])
        elif len(descriptors) > n_keypoints:
            descriptors = descriptors[:n_keypoints]
        
        return descriptors.flatten()
    
    def extraer_caracteristicas_hog(self, imagen):
        """Extrae características HOG"""
        if imagen.max() <= 1.0:
            imagen = (imagen * 255).astype(np.uint8)
        else:
            imagen = imagen.astype(np.uint8)
        
        win_size = (imagen.shape[1] // 16 * 16, imagen.shape[0] // 16 * 16)
        img_resized = cv2.resize(imagen, win_size)
        
        hog = cv2.HOGDescriptor(
            win_size,
            (16, 16),
            (8, 8),
            (8, 8),
            9
        )
        
        descriptor = hog.compute(img_resized)
        return descriptor.flatten()
    
    def extraer_caracteristicas(self, imagen):
        """Extrae características según el método configurado"""
        if self.metodo == 'cnn':
            return self.extraer_caracteristicas_cnn(imagen)
        elif self.metodo == 'hu_moments':
            return self.extraer_caracteristicas_hu_moments(imagen)
        elif self.metodo == 'sift':
            return self.extraer_caracteristicas_sift(imagen)
        elif self.metodo == 'hog':
            return self.extraer_caracteristicas_hog(imagen)
        else:
            raise ValueError(f"Método desconocido: {self.metodo}")
    
    def normalizar_l2(self, features):
        """
        Normalización L2 por punto (independiente de otros puntos).
        NO requiere estadísticas globales como StandardScaler.
        """
        features = np.asarray(features).astype(float)
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features
    
    # ==================== CLUSTERING ONLINE ====================
    
    def _calcular_metricas(self, X_scaled, predicciones):
        """Calcula métricas de validación interna y externa"""
        metricas = {
            'internas': {},
            'externas': {}
        }
        
        # Métricas internas (siempre disponibles)
        try:
            if len(np.unique(predicciones)) > 1:
                metricas['internas']['silhouette'] = float(silhouette_score(X_scaled, predicciones))
                metricas['internas']['dunn'] = float(dunn_index(X_scaled, predicciones))
            else:
                metricas['internas']['silhouette'] = 0.0
                metricas['internas']['dunn'] = 0.0
        except Exception as e:
            print(f"⚠ Error calculando métricas internas: {e}")
            metricas['internas']['silhouette'] = 0.0
            metricas['internas']['dunn'] = 0.0
        
        # Métricas externas (solo si hay etiquetas ground truth)
        labels_reales = [l for l in self.labels_accumulated if l is not None]
        if labels_reales:
            try:
                # Filtrar solo las muestras con etiqueta
                indices_con_label = [i for i, l in enumerate(self.labels_accumulated) if l is not None]
                predicciones_filtradas = predicciones[indices_con_label]
                labels_filtradas = np.array(labels_reales)
                
                metricas['externas']['ari'] = float(adjusted_rand_score(labels_filtradas, predicciones_filtradas))
                metricas['externas']['nmi'] = float(normalized_mutual_info_score(labels_filtradas, predicciones_filtradas))
                metricas['externas']['ami'] = float(adjusted_mutual_info_score(labels_filtradas, predicciones_filtradas))
                metricas['externas']['n_etiquetadas'] = len(labels_reales)
            except Exception as e:
                print(f"⚠ Error calculando métricas externas: {e}")
        
        return metricas
    
    def obtener_distribucion_clusters(self):
        """Retorna la distribución actual de imágenes por cluster"""
        if not self.has_data or len(self.predictions_accumulated) == 0:
            return [0] * self.n_clusters
        
        # Contar cuántas imágenes hay en cada cluster
        distribucion = [0] * self.n_clusters
        for pred in self.predictions_accumulated:
            if 0 <= pred < self.n_clusters:
                distribucion[pred] += 1
        
        return distribucion
    
    def calcular_metricas(self):
        """Calcula métricas del estado actual del sistema"""
        if not self.has_data:
            return {'internas': {}, 'externas': {}}
        
        # Normalización L2 (sin scaler global)
        X_norm = np.array([self.normalizar_l2(x) for x in self.X_accumulated])
        predicciones = np.array(self.predictions_accumulated)
        
        return self._calcular_metricas(X_norm, predicciones)
    
    def reset_sistema(self):
        """Reinicia el sistema eliminando todos los datos acumulados de memoria"""
        self.model = None
        # SIN scaler global, SIN PCA
        self.X_accumulated = []
        self.labels_accumulated = []
        self.filenames_accumulated = []
        self.predictions_accumulated = []
        self.has_data = False
        self._history_ingested = False
        
        print("✓ Sistema reiniciado - datos borrados de memoria")
    
    # ==================== PROCESAMIENTO STREAMING (ONLINE PURO) ====================
    
    def procesar_imagenes_streaming(self, imagenes_data, realtime=False):
        """
        GENERADOR ONLINE PURO: procesa cada imagen COMPLETAMENTE antes de pasar a la siguiente.
        """
        total = len(imagenes_data)
        print(f"\n[STREAMING ONLINE PURO] Iniciando procesamiento de {total} imágenes... realtime={realtime}")
        
        # Evento de inicio
        yield {
            'type': 'start',
            'total': total,
            'metodo': self.metodo,
            'n_clusters': self.n_clusters
        }
        
        # =====================================================
        # Configurar modelo con capacidades
        # =====================================================
        n_samples_total = len(self.X_accumulated) + total
        
        if self.cluster_sizes:
            if (not realtime) and sum(self.cluster_sizes) != n_samples_total:
                yield {
                    'type': 'error',
                    'error': f'La suma de tamaños ({sum(self.cluster_sizes)}) no coincide con el total ({n_samples_total})'
                }
                return
            capacities = self.cluster_sizes
        else:
            base = n_samples_total // self.n_clusters
            resto = n_samples_total % self.n_clusters
            capacities = [base + (1 if i < resto else 0) for i in range(self.n_clusters)]
        
        print(f"[STREAMING] Capacidades: {capacities}")
        
        # Crear modelo si no existe
        with self.lock:
            if self.model is None:
                if realtime:
                    self.model = OnlineDynamicClustering(
                        max_k=self.n_clusters,
                        capacities=capacities,
                        threshold=self.distance_threshold  # <--- CAMBIO: Pasar umbral
                    )
                else:
                    self.model = OnlineKMeansCapacitado(
                        k=self.n_clusters, 
                        capacities=capacities,
                        threshold=self.distance_threshold  # <--- CAMBIO: Pasar umbral
                    )
                self._history_ingested = False

        # Re-procesar datos previos (solo si no es realtime y solo una vez)
        if (not realtime) and self.has_data and len(self.X_accumulated) > 0 and (not self._history_ingested):
            print(f"[STREAMING] Re-procesando {len(self.X_accumulated)} datos previos...")
            with self.lock:
                self.predictions_accumulated = []
                if isinstance(self.model, OnlineDynamicClustering):
                    self.model.centers = []
                    self.model.remaining = capacities[:]
                    self.model.counts = []
                    self.model.cluster_centers_ = None
                else:
                    self.model.reset()
                for x_raw in self.X_accumulated:
                    x_norm = self.normalizar_l2(x_raw)
                    pred = self.model.partial_fit(x_norm)
                    self.predictions_accumulated.append(int(pred))
                self._history_ingested = True
        
        # =====================================================
        # PROCESAMIENTO ONLINE PURO - imagen por imagen
        # =====================================================
        print(f"[STREAMING] Procesando {total} nuevas imágenes ONLINE...")
        
        imagenes_procesadas = 0
        
        for idx, img_data in enumerate(imagenes_data):
            try:
                # 1. OBTENER DATOS DE LA IMAGEN
                if 'bytes' in img_data:
                    img_input = img_data['bytes']
                    img_bytes = img_data['bytes']
                    filename = img_data['filename']
                elif 'path' in img_data:
                    img_input = img_data['path']
                    filename = Path(img_data['path']).name
                    with open(img_data['path'], 'rb') as f:
                        img_bytes = f.read()
                else:
                    continue
                
                label = img_data.get('label', None)
                
                # 2. EXTRAER CARACTERÍSTICAS (de esta imagen solamente)
                img_prep = self.preprocesar_imagen(img_input)
                features_raw = self.extraer_caracteristicas(img_prep)
                
                # 3. NORMALIZAR L2 (independiente - no necesita otras imágenes)
                features_norm = self.normalizar_l2(features_raw)

                with self.lock:
                    # 4. CLUSTERING ONLINE INMEDIATO
                    cluster = self.model.partial_fit(features_norm)

                    # 5. CALCULAR DISTANCIA AL CENTROIDE
                    dist = np.linalg.norm(features_norm - self.model.cluster_centers_[cluster])

                    # 6. ACTUALIZAR ACUMULADOS
                    self.X_accumulated.append(features_raw)
                    self.labels_accumulated.append(label)
                    self.filenames_accumulated.append(filename)
                    self.predictions_accumulated.append(int(cluster))
                    self.has_data = True
                
                imagenes_procesadas += 1
                
                # Métricas internas en tiempo real (sin externas)
                metricas_internas = None
                if realtime:
                    with self.lock:
                        X_copy = list(self.X_accumulated)
                        pred_copy = list(self.predictions_accumulated)
                    X_all_norm = np.array([self.normalizar_l2(x) for x in X_copy])
                    metricas_internas = self._calcular_metricas(X_all_norm, np.array(pred_copy))
                    metricas_internas['externas'] = {}

                # 7. EMITIR RESULTADO INMEDIATAMENTE
                yield {
                    'type': 'progress',
                    'current': idx + 1,
                    'total': total,
                    'resultado': {
                        'nombre': filename,
                        'image_bytes': img_bytes,
                        'cluster': int(cluster),
                        'distancia': float(dist),
                        'confianza': float(1.0 / (1.0 + dist)),
                        'label_real': label
                    },
                    'cluster_distribution': self.model.get_distribution(),
                    'metricas': metricas_internas
                }
                
                print(f"  [{idx+1}/{total}] {filename} -> Cluster {cluster}")
                
            except Exception as e:
                print(f"  Error en {img_data.get('filename', idx)}: {e}")
                yield {
                    'type': 'image_error',
                    'current': idx + 1,
                    'total': total,
                    'filename': img_data.get('filename', f'imagen_{idx}'),
                    'error': str(e)
                }
                continue
        
        # =====================================================
        # CALCULAR MÉTRICAS FINALES
        # =====================================================
        print("[STREAMING] Calculando métricas finales...")
        
        try:
            X_all_norm = np.array([self.normalizar_l2(x) for x in self.X_accumulated])
            predicciones = np.array(self.predictions_accumulated)
            metricas = self._calcular_metricas(X_all_norm, predicciones)
        except Exception as e:
            print(f"  Error métricas: {e}")
            metricas = {'internas': {}, 'externas': {}}
        
        print(f"[STREAMING] Completado! Total acumulado: {len(self.X_accumulated)}")
        
        # Evento de finalización
        yield {
            'type': 'complete',
            'total': imagenes_procesadas,
            'metricas': metricas,
            'total_acumulado': len(self.X_accumulated),
            'distribucion_final': self.model.get_distribution() if self.model else []
        }


if __name__ == "__main__":
    print("Sistema de Clustering Online Incremental")
    print("Para usar desde la interfaz web, ejecuta: python app.py")