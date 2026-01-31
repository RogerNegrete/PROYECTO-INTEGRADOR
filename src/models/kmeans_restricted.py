import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, pdist


class KMeansRestricted:
    """
    K-Means con restricciones de tamaño de cluster.
    Asegura que cada cluster tenga un tamaño mínimo y máximo.
    """
    
    def __init__(self, n_clusters, min_size=None, max_size=None, max_iter=300, random_state=42):
        """
        Args:
            n_clusters: Número de clusters
            min_size: Tamaño mínimo por cluster (None = sin restricción mínima)
            max_size: Tamaño máximo por cluster (None = sin restricción máxima)
            max_iter: Número máximo de iteraciones
            random_state: Semilla aleatoria
        """
        self.n_clusters = n_clusters
        self.min_size = min_size
        self.max_size = max_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def _inicializar_centroides(self, X):
        """Inicializa centroides usando k-means++"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Primer centroide aleatorio
        centroides = [X[np.random.randint(n_samples)]]
        
        # Resto de centroides con k-means++
        for _ in range(1, self.n_clusters):
            # Calcular distancias mínimas a centroides existentes
            distancias = np.min([np.sum((X - c)**2, axis=1) for c in centroides], axis=0)
            # Probabilidades proporcionales a distancias al cuadrado
            probs = distancias / distancias.sum()
            # Seleccionar siguiente centroide
            idx = np.random.choice(n_samples, p=probs)
            centroides.append(X[idx])
            
        return np.array(centroides)
    
    def _asignar_con_restricciones(self, X, centroides):
        """
        Asigna puntos a clusters respetando restricciones de tamaño.
        """
        n_samples = X.shape[0]
        
        # Calcular distancias de cada punto a cada centroide (optimizado)
        distancias = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            diff = X - centroides[i]
            distancias[:, i] = np.einsum('ij,ij->i', diff, diff)
        
        # Inicializar asignaciones
        labels = np.full(n_samples, -1)
        
        # Tamaños por defecto si no se especifican
        min_size = self.min_size if self.min_size else 0
        max_size = self.max_size if self.max_size else n_samples
        
        # Contar asignaciones por cluster
        cluster_counts = np.zeros(self.n_clusters, dtype=int)
        
        # Ordenar puntos por distancia mínima a cualquier centroide (más difíciles primero)
        min_distancias = np.min(distancias, axis=1)
        orden = np.argsort(-min_distancias)
        
        # Asignar puntos ordenados
        for idx in orden:
            # Encontrar clusters disponibles (que no hayan alcanzado max_size)
            clusters_disponibles = np.where(cluster_counts < max_size)[0]
            
            if len(clusters_disponibles) == 0:
                # Si no hay clusters disponibles, forzar asignación al más cercano
                cluster = np.argmin(distancias[idx])
            else:
                # Asignar al cluster más cercano entre los disponibles
                distancias_disponibles = distancias[idx, clusters_disponibles]
                cluster = clusters_disponibles[np.argmin(distancias_disponibles)]
            
            labels[idx] = cluster
            cluster_counts[cluster] += 1
        
        # Verificar restricción mínima y redistribuir si es necesario
        if self.min_size:
            for cluster_id in range(self.n_clusters):
                if cluster_counts[cluster_id] < min_size:
                    # Necesita más puntos
                    faltantes = min_size - cluster_counts[cluster_id]
                    
                    # Buscar puntos en clusters con exceso
                    for _ in range(faltantes):
                        # Encontrar clusters con más del mínimo
                        clusters_con_exceso = np.where(cluster_counts > min_size)[0]
                        if len(clusters_con_exceso) == 0:
                            break
                        
                        # Buscar el punto más cercano a este cluster en otros clusters
                        otros_clusters = labels != cluster_id
                        if not np.any(otros_clusters):
                            break
                            
                        distancias_otros = distancias[otros_clusters, cluster_id]
                        idx_relativo = np.argmin(distancias_otros)
                        idx_absoluto = np.where(otros_clusters)[0][idx_relativo]
                        
                        # Reasignar punto
                        cluster_anterior = labels[idx_absoluto]
                        labels[idx_absoluto] = cluster_id
                        cluster_counts[cluster_id] += 1
                        cluster_counts[cluster_anterior] -= 1
        
        return labels
    
    def _actualizar_centroides(self, X, labels):
        """Actualiza centroides como la media de los puntos asignados"""
        centroides = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            puntos_cluster = X[labels == k]
            if len(puntos_cluster) > 0:
                centroides[k] = puntos_cluster.mean(axis=0)
        return centroides
    
    def _calcular_inercia(self, X, labels, centroides):
        """Calcula la suma de distancias al cuadrado dentro de los clusters"""
        inercia = 0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                diff = X[mask] - centroides[k]
                inercia += np.sum(diff * diff)
        return inercia
    
    def fit(self, X):
        """
        Entrena el modelo K-Means con restricciones.
        """
        print(f"\n{'='*60}")
        print(f"K-Means con restricciones de tamaño")
        print(f"Clusters: {self.n_clusters}")
        print(f"Min size: {self.min_size}, Max size: {self.max_size}")
        print(f"{'='*60}")
        
        # Inicializar centroides
        self.centroids_ = self._inicializar_centroides(X)
        
        # Iteraciones
        for iteracion in range(self.max_iter):
            # Asignar puntos con restricciones
            labels_nuevos = self._asignar_con_restricciones(X, self.centroids_)
            
            # Actualizar centroides
            centroides_nuevos = self._actualizar_centroides(X, labels_nuevos)
            
            # Verificar convergencia
            if np.allclose(self.centroids_, centroides_nuevos):
                print(f"✓ Convergencia alcanzada en iteración {iteracion + 1}")
                break
                
            self.centroids_ = centroides_nuevos
            self.labels_ = labels_nuevos
            
            if (iteracion + 1) % 50 == 0:
                print(f"Iteración {iteracion + 1}/{self.max_iter}")
        
        self.labels_ = labels_nuevos
        self.inertia_ = self._calcular_inercia(X, self.labels_, self.centroids_)
        
        # Mostrar tamaños de clusters
        unique, counts = np.unique(self.labels_, return_counts=True)
        print(f"\nTamaños de clusters: {dict(zip(unique, counts))}")
        
        return self
    
    def predict(self, X):
        """Predice el cluster para nuevos datos"""
        distancias = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            diff = X - self.centroids_[i]
            distancias[:, i] = np.einsum('ij,ij->i', diff, diff)
        return np.argmin(distancias, axis=1)


def cargar_dataset(ruta_csv):
    """
    Carga un dataset CSV.
    
    Returns:
        tuple: (features, labels) donde labels son las clases verdaderas si existen
    """
    print(f"Cargando {os.path.basename(ruta_csv)}...")
    
    # Cargar con dtype float64 para evitar problemas de conversión
    try:
        df = pd.read_csv(ruta_csv, low_memory=False, dtype={col: np.float64 for col in range(100000)}, on_bad_lines='skip')
    except:
        # Si falla, cargar línea por línea puede ser necesario, pero primero intentar más simple
        df = pd.read_csv(ruta_csv, engine='python', on_bad_lines='skip')
    
    print(f"✓ Archivo cargado: {df.shape}")
    
    # Separar características de etiquetas
    if 'clase' in df.columns:
        # Si tiene columna 'archivo', es un ground truth
        if 'archivo' in df.columns:
            # Solo retornar las clases, sin las características
            return None, df['clase'].values
        else:
            # Es un archivo de características con clase
            X = df.drop('clase', axis=1).values.astype(np.float64)
            y = df['clase'].values
            return X, y
    else:
        return df.values.astype(np.float64), None


def calcular_indice_dunn(X, labels):
    """
    Calcula el índice de Dunn para evaluar clustering.
    Dunn = min(distancia entre clusters) / max(diámetro de clusters)
    Valores más altos son mejores.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # Calcular centroides de cada cluster
    centroides = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
    
    # Distancia mínima entre centroides (distancia inter-cluster)
    distancias_inter = pdist(centroides)
    min_distancia_inter = np.min(distancias_inter) if len(distancias_inter) > 0 else 0
    
    # Diámetro máximo de clusters (distancia intra-cluster)
    max_diametro = 0
    for k in unique_labels:
        cluster_points = X[labels == k]
        if len(cluster_points) > 1:
            diametro = np.max(pdist(cluster_points))
            max_diametro = max(max_diametro, diametro)
    
    # Evitar división por cero
    if max_diametro == 0:
        return 0.0
    
    return min_distancia_inter / max_diametro


def evaluar_clustering(X, labels_pred, labels_true=None):
    """
    Evalúa el clustering con métricas internas y externas.
    """
    metricas = {}
    
    # Métricas internas (no requieren etiquetas verdaderas)
    if len(np.unique(labels_pred)) > 1:
        metricas['silhouette'] = silhouette_score(X, labels_pred)
        metricas['dunn'] = calcular_indice_dunn(X, labels_pred)
    
    # Métricas externas (requieren etiquetas verdaderas)
    if labels_true is not None:
        metricas['ari'] = adjusted_rand_score(labels_true, labels_pred)
        metricas['nmi'] = normalized_mutual_info_score(labels_true, labels_pred)
        metricas['ami'] = adjusted_mutual_info_score(labels_true, labels_pred)
    
    return metricas


def visualizar_resultados(resultados, nombre_dataset):
    """
    Visualiza los resultados del clustering.
    """
    # Crear directorio para resultados
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', '..', 'results', 'kmeans_restricted')
    os.makedirs(results_dir, exist_ok=True)
    
    # Obtener las métricas del único método
    metodo = list(resultados.keys())[0]
    metricas = resultados[metodo]['metricas']
    
    # Crear figura con subplots para las métricas
    metricas_nombres = [m for m in ['silhouette', 'dunn', 'ari', 'nmi', 'ami'] if m in metricas]
    n_metricas = len(metricas_nombres)
    
    fig, axes = plt.subplots(1, n_metricas, figsize=(5*n_metricas, 5))
    fig.suptitle(f'Métricas K-Means con Restricciones - {nombre_dataset}', fontsize=16)
    
    if n_metricas == 1:
        axes = [axes]
    
    for idx, metrica in enumerate(metricas_nombres):
        ax = axes[idx]
        valor = metricas[metrica]
        
        bar = ax.bar([metrica], [valor], color='#2ca02c', width=0.5)
        ax.set_title(f'{metrica.upper()}')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([min(0, valor * 1.1), max(1, valor * 1.1)])
        
        # Agregar valor encima de la barra
        ax.text(0, valor, f'{valor:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    ruta_fig = os.path.join(results_dir, f'{nombre_dataset}_metricas.png')
    plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {ruta_fig}")
    plt.close()
    
    # Distribución de clusters
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Distribución de Clusters - {nombre_dataset}', fontsize=16)
    
    labels = resultados[metodo]['labels']
    unique, counts = np.unique(labels, return_counts=True)
    
    bars = ax.bar(unique, counts, color='steelblue')
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Número de muestras', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores encima de cada barra
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    ruta_fig = os.path.join(results_dir, f'{nombre_dataset}_distribucion.png')
    plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {ruta_fig}")
    plt.close()


def procesar_dataset(ruta_csv, nombre_dataset, labels_true_csv=None):
    """
    Procesa un dataset con K-means con diferentes configuraciones.
    
    Args:
        ruta_csv: Ruta al archivo CSV con características
        nombre_dataset: Nombre del dataset
        labels_true_csv: Ruta al CSV con etiquetas verdaderas (ground truth)
    """
    print(f"\n{'='*70}")
    print(f"PROCESANDO: {nombre_dataset}")
    print(f"{'='*70}")
    
    # Cargar dataset
    X, _ = cargar_dataset(ruta_csv)
    print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # Cargar etiquetas verdaderas si existen
    labels_true = None
    if labels_true_csv and os.path.exists(labels_true_csv):
        _, labels_true = cargar_dataset(labels_true_csv)
        n_clases = len(np.unique(labels_true))
        print(f"Etiquetas verdaderas cargadas: {n_clases} clases")
    else:
        # Inferir número de clusters del nombre
        if 'Cats_Dogs_Foxes' in nombre_dataset:
            n_clases = 3
        elif 'Vocals' in nombre_dataset:
            n_clases = 5
        else:
            n_clases = 3  # Por defecto
        print(f"Número de clusters inferido: {n_clases}")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Datos normalizados")
    
    # Aplicar PCA si hay demasiadas características (optimización)
    if X_scaled.shape[1] > 1000:
        print(f"Reduciendo dimensionalidad de {X_scaled.shape[1]} características...")
        n_components = min(200, X_scaled.shape[0] - 1)  # Reducido a 200 para más velocidad
        pca = PCA(n_components=n_components, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        varianza_explicada = pca.explained_variance_ratio_.sum()
        print(f"✓ PCA aplicado: {X_scaled.shape[1]} componentes (varianza: {varianza_explicada:.2%})")
    elif X_scaled.shape[1] > 100:
        print(f"Reduciendo dimensionalidad de {X_scaled.shape[1]} características...")
        n_components = min(100, X_scaled.shape[0] - 1)  # Reducir incluso datasets medianos
        pca = PCA(n_components=n_components, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        varianza_explicada = pca.explained_variance_ratio_.sum()
        print(f"✓ PCA aplicado: {X_scaled.shape[1]} componentes (varianza: {varianza_explicada:.2%})")
    else:
        print(f"Dimensionalidad manejable: {X_scaled.shape[1]} características")
    
    # Calcular restricciones
    n_samples = X.shape[0]
    tamaño_ideal = n_samples // n_clases
    
    resultados = {}
    
    # K-Means con restricciones min y max
    min_size = int(tamaño_ideal * 0.5)  # Mínimo 50% del tamaño ideal
    max_size = int(tamaño_ideal * 1.5)  # Máximo 150% del tamaño ideal
    print(f"\n--- K-Means CON restricciones MIN y MAX (min={min_size}, max={max_size}) ---")
    kmeans_minmax = KMeansRestricted(n_clusters=n_clases, min_size=min_size, max_size=max_size, random_state=42)
    kmeans_minmax.fit(X_scaled)
    unique_minmax, counts_minmax = np.unique(kmeans_minmax.labels_, return_counts=True)
    print(f"Clusters creados: {len(unique_minmax)}")
    print(f"Tamaño de cada cluster: {dict(zip(unique_minmax, counts_minmax))}")
    metricas_minmax = evaluar_clustering(X_scaled, kmeans_minmax.labels_, labels_true)
    resultados['Con Min y Max Size'] = {
        'labels': kmeans_minmax.labels_,
        'metricas': metricas_minmax,
        'inercia': kmeans_minmax.inertia_
    }
    
    # Mostrar resumen de métricas
    print(f"\n{'='*70}")
    print("RESUMEN DE MÉTRICAS")
    print(f"{'='*70}")
    
    for metodo, datos in resultados.items():
        print(f"\n{metodo}:")
        print(f"  Inercia: {datos['inercia']:.2f}")
        for metrica, valor in datos['metricas'].items():
            print(f"  {metrica}: {valor:.4f}")
    
    # Visualizar resultados
    visualizar_resultados(resultados, nombre_dataset)
    
    # Guardar resultados en CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', '..', 'results', 'kmeans_restricted')
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar etiquetas predichas
    for metodo, datos in resultados.items():
        metodo_safe = metodo.replace(' ', '_').lower()
        df_resultado = pd.DataFrame({
            'cluster_predicho': datos['labels']
        })
        if labels_true is not None:
            df_resultado['clase_verdadera'] = labels_true
        
        ruta_salida = os.path.join(results_dir, f'{nombre_dataset}_{metodo_safe}.csv')
        df_resultado.to_csv(ruta_salida, index=False)
        print(f"✓ Resultados guardados: {ruta_salida}")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETADO: {nombre_dataset}")
    print(f"{'='*70}\n")
    
    return resultados


def main():
    """
    Función principal que procesa todos los datasets con K-Means restringido.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(script_dir, '..', 'features')
    
    print("\n" + "="*70)
    print("K-MEANS CON RESTRICCIONES DE TAMAÑO")
    print("="*70)
    
    # Datasets a procesar - AHORA INCLUYENDO CNN
    datasets = [
        # Cats_Dogs_Foxes
        ('Cats_Dogs_Foxes_hog.csv', 'Cats_Dogs_Foxes_HOG', 'Cats_Dogs_Foxes_ground_true.csv'),
        ('Cats_Dogs_Foxes_sift.csv', 'Cats_Dogs_Foxes_SIFT', 'Cats_Dogs_Foxes_ground_true.csv'),
        ('Cats_Dogs_Foxes_hu_moments.csv', 'Cats_Dogs_Foxes_HuMoments', 'Cats_Dogs_Foxes_ground_true.csv'),
        ('Cats_Dogs_Foxes_cnn.csv', 'Cats_Dogs_Foxes_CNN', 'Cats_Dogs_Foxes_ground_true.csv'),
        
        # Vocals_UPIIT2025
        ('Vocals_UPIIT2025_hog.csv', 'Vocals_UPIIT2025_HOG', 'Vocals_UPIIT2025_ground_true.csv'),
        ('Vocals_UPIIT2025_sift.csv', 'Vocals_UPIIT2025_SIFT', 'Vocals_UPIIT2025_ground_true.csv'),
        ('Vocals_UPIIT2025_hu_moments.csv', 'Vocals_UPIIT2025_HuMoments', 'Vocals_UPIIT2025_ground_true.csv'),
        ('Vocals_UPIIT2025_cnn.csv', 'Vocals_UPIIT2025_CNN', 'Vocals_UPIIT2025_ground_true.csv'),
    ]
    
    # Procesar cada dataset
    for archivo, nombre, ground_truth in datasets:
        ruta_csv = os.path.join(features_dir, archivo)
        ruta_ground_truth = os.path.join(features_dir, ground_truth)
        
        if os.path.exists(ruta_csv):
            procesar_dataset(ruta_csv, nombre, ruta_ground_truth)
        else:
            print(f"⚠ ADVERTENCIA: No se encontró {ruta_csv}")
    
    print("\n" + "="*70)
    print("✓ PROCESO COMPLETO FINALIZADO")
    print("="*70)


if __name__ == "__main__":
    main()

