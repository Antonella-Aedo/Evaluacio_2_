"""Archivo: src/unsupervised_learning.py

Helpers para aprendizaje no supervisado: clustering, reducción de dimensionalidad y trazado.

Funciones reutilizables desde notebooks que mantienen reproducibilidad mediante
parámetros `random_state`. Realizan comprobaciones básicas y guardan plots en
`results/plots/`.
"""
from typing import Tuple, Any, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# Función: `ensure_dir`
# Qué hace: crear el directorio si no existe.
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Función: `preprocess_for_unsupervised`
# Qué hace: preparar características para aprendizaje no supervisado (codificar, eliminar columnas, escalar).
def preprocess_for_unsupervised(df: pd.DataFrame, drop_columns: Optional[list] = None,
                                encode: bool = True, scale: bool = True) -> Tuple[np.ndarray, Any]:
    """Preparar features para aprendizaje no supervisado.

    - Opcionalmente elimina columnas indicadas en `drop_columns`.
    - Codifica categóricas con one-hot (mantiene compatibilidad).
    - Aplica `StandardScaler` si `scale` es True.

    Devuelve (X_array, scaler_or_none).
    """
    if drop_columns is None:
        drop_columns = []
    X = df.copy()
    X = X.drop(columns=[c for c in drop_columns if c in X.columns], errors='ignore')
    if encode:
        X = pd.get_dummies(X, drop_first=True)
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return X.values, None


# Función: `perform_pca`
# Qué hace: calcular PCA y devolver el objeto PCA ajustado y los datos transformados.
def perform_pca(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> Tuple[Any, np.ndarray]:
    """Calcular PCA y devolver PCA ajustado y datos transformados."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


# Función: `perform_tsne`
# Qué hace: calcular embedding t-SNE (aleatorio; usar random_state para reproducibilidad).
def perform_tsne(X: np.ndarray, n_components: int = 2, random_state: int = 42, perplexity: int = 30) -> np.ndarray:
    """Calcular embedding t-SNE. Para datasets grandes considere submuestreo.

    t-SNE es estocástico; `random_state` asegura reproducibilidad.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    return X_tsne


# Función: `cluster_kmeans`
# Qué hace: ejecutar KMeans y devolver modelo, etiquetas y métricas (inercia, silhouette).
def cluster_kmeans(X: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[KMeans, np.ndarray, dict]:
    """Ejecutar KMeans y devolver modelo, labels y métricas (inertia, silhouette)."""
    k = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = k.fit_predict(X)
    metrics = {"inertia": float(k.inertia_)}
    try:
        if len(set(labels)) > 1 and len(labels) > len(set(labels)):
            metrics["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        metrics["silhouette"] = None
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics["davies_bouldin"] = None
    return k, labels, metrics


# Función: `cluster_dbscan`
# Qué hace: ejecutar DBSCAN y devolver modelo, etiquetas y métricas cuando aplica.
def cluster_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[DBSCAN, np.ndarray, dict]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    metrics = {}
    try:
        if len(set(labels)) - (1 if -1 in labels else 0) > 1:
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
    return db, labels, metrics


# Función: `cluster_agglomerative`
# Qué hace: ejecutar clustering aglomerativo y devolver modelo, etiquetas y métricas.
def cluster_agglomerative(X: np.ndarray, n_clusters: int = 3) -> Tuple[AgglomerativeClustering, np.ndarray, dict]:
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(X)
    metrics = {}
    try:
        if len(set(labels)) > 1:
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
    return agg, labels, metrics


# Función: `plot_clusters_2d`
# Qué hace: trazar en 2D la representación coloreada por clusters y guardar la imagen.
def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, title: str, save_path: str, xlabel: str = 'Dim 1', ylabel: str = 'Dim 2') -> None:
    """Scatter plot de la representación 2D coloreada por etiquetas de cluster y guardada en disco."""
    ensure_dir(os.path.dirname(save_path) or '.')
    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        label_name = f'Cluster {lab}' if lab != -1 else 'Noise'
        plt.scatter(X_2d[mask,0], X_2d[mask,1], s=40, color=colors(idx), label=label_name, alpha=0.7, edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
