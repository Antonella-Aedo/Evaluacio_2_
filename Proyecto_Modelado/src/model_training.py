"""Archivo: src/model_training.py

Ayudantes para entrenamiento de modelos: wrappers de entrenamiento y persistencia.

Este módulo ofrece helpers para construir `Pipeline`s de sklearn para
clasificación y regresión, entrenarlos y guardar los modelos resultantes.
Las pipelines garantizan que el preprocesamiento (escalado) sea consistente.
"""
from typing import Dict, Any, Tuple
import joblib
import json
from pathlib import Path
from datetime import datetime
import platform
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Función: `train_model`
# Qué hace: ajusta `model` usando X_train/y_train y devuelve el estimador entrenado.
def train_model(model: Any, X_train, y_train) -> Any:
    """Ajustar `model` en X_train/y_train y devolver el estimador entrenado.

    Wrapper que centraliza manejo de errores.
    """
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Error entrenando el modelo: {e}")
    return model


# Función: `create_classification_pipelines`
# Qué hace: construir pipelines listas para entrenar en tareas de clasificación.
def create_classification_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Devolver un diccionario de pipelines listas para entrenar (clasificación).

    - `LogisticRegression` con escalado: baseline lineal.
    - `RandomForestClassifier`: ensemble no lineal robusto.

    Usar pipelines asegura el mismo preprocesamiento en CV y predicción.
    """
    pipelines: Dict[str, Pipeline] = {}
    # Baseline linear model - interpretable, fast
    pipelines["logistic"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=random_state, max_iter=1000))
    ])

    # Stronger non-linear baseline - handles feature interactions
    pipelines["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=random_state))
    ])

    return pipelines


# Función: `create_regression_pipelines`
# Qué hace: construir pipelines para tareas de regresión.
def create_regression_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Devolver un diccionario de pipelines para regresión.

    - `LinearRegression` con escalado: baseline lineal.
    - `RandomForestRegressor`: opción no lineal.
    """
    pipelines: Dict[str, Pipeline] = {}
    pipelines["linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    pipelines["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(random_state=random_state))
    ])

    return pipelines


# Función: `train_and_predict_classifiers`
# Qué hace: entrenar varios clasificadores y devolver modelos entrenados y predicciones.
def train_and_predict_classifiers(models: Dict[str, Any], X_train, X_test, y_train) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Entrenar múltiples modelos clasificadores y devolver (modelos_entrenados, predicciones).

    `predicciones` es un dict name->y_pred.
    """
    trained = {}
    preds = {}
    for name, model in models.items():
        trained[name] = train_model(model, X_train, y_train)
        try:
            preds[name] = trained[name].predict(X_test)
        except Exception:
            preds[name] = None
    return trained, preds


# Función: `save_model`
# Qué hace: persistir un modelo entrenado en disco usando joblib.
def save_model(model: Any, path: str) -> None:
    """Guardar un modelo entrenado en disco usando joblib."""
    try:
        joblib.dump(model, path)
    except Exception as e:
        raise IOError(f"No se pudo guardar el modelo en {path}: {e}")


# Función: `guardar_modelo_con_metadata`
# Qué hace: guarda un modelo serializado con joblib y crea un archivo JSON
# con metadatos (parámetros, métricas, fecha, versiones).
# Propósito: facilitar el almacenamiento reproducible de modelos en
# `models/trained_models/` junto a su metadata para auditoría.
def guardar_modelo_con_metadata(model: Any, ruta_salida: str, metadata: Dict[str, Any]) -> None:
    """Guardar modelo y metadata.

    - `ruta_salida`: ruta completa al archivo de salida (p.ej. "models/trained_models/rfc__2026-05-10T14-30__v1.joblib").
    - `metadata`: diccionario con campos como `model_name`, `version`, `params`, `metrics`, `data_source`.

    Se crea además un archivo JSON con el mismo nombre y extensión `.json` conteniendo la metadata extendida.
    """
    p = Path(ruta_salida)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Guardar el modelo serializado
    try:
        joblib.dump(model, str(p))
    except Exception as e:
        raise IOError(f"Error guardando el modelo en {p}: {e}")

    # Completar metadata mínima
    metadata.setdefault("date", datetime.utcnow().isoformat())
    metadata.setdefault("python_version", platform.python_version())
    metadata.setdefault("sklearn_version", sklearn.__version__)

    meta_path = p.with_suffix('.json')
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise IOError(f"Error guardando metadata en {meta_path}: {e}")


# Función: `cargar_modelo_con_metadata`
# Qué hace: carga un modelo serializado y su archivo JSON de metadatos (si existe).
# Retorna: (modelo, metadata_dict)
def cargar_modelo_con_metadata(ruta: str) -> Tuple[Any, Dict[str, Any]]:
    """Cargar modelo y metadata asociada.

    - `ruta`: ruta al archivo serializado (.joblib/.pkl).
    - Devuelve una tupla: (modelo, metadata_dict). Si no existe el JSON, `metadata_dict` será vacío.
    """
    p = Path(ruta)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de modelo: {p}")
    try:
        modelo = joblib.load(str(p))
    except Exception as e:
        raise IOError(f"Error cargando el modelo desde {p}: {e}")

    meta = {}
    meta_path = p.with_suffix('.json')
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    return modelo, meta
