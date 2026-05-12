"""Archivo: src/model_evaluation.py

Utilidades de evaluación para clasificación y regresión.

Este módulo añade helpers para validación cruzada, trazado y guardado
de métricas para evaluación consistente de modelos siguiendo buenas prácticas.
"""
from typing import Dict, Any, Iterable, Optional
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _wrap_in_pipeline_if_needed(estimator):
    """Si `estimator` no es un `Pipeline`, envolverlo en uno que incluya
    un `StandardScaler` seguido del estimador. Esto garantiza que el
    preprocesamiento sea consistente en CV y predicción.
    """
    if isinstance(estimator, Pipeline):
        return estimator
    return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])


def _project_root() -> str:
    """Determinar la raíz del proyecto (carpeta que contiene `src`)."""
    module_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(module_dir, os.pardir))


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Convertir una ruta relativa en absoluta usando la raíz del proyecto.

    Si `path` ya es absoluta, se devuelve tal cual. Si `path` es None,
    devuelve None.
    """
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(_project_root(), path)


# Función: `classification_metrics`
# Qué hace: calcula métricas comunes de clasificación y devuelve un diccionario.
def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calcular métricas comunes de clasificación y devolver un dict.

    Devuelve accuracy, precision, recall y f1. Maneja divisiones por cero.
    """
    if y_pred is None:
        return {}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# Función: `regression_metrics`
# Qué hace: calcula MAE, MSE, RMSE y R2 para regresión y devuelve un diccionario.
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calcular MAE, MSE, RMSE y R2 para regresión y devolver un dict."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# Función: `cross_validate_models_classification`
# Qué hace: ejecutar validación cruzada para múltiples modelos de clasificación
def cross_validate_models_classification(models: Dict[str, Any], X, y,
                                         cv_splits: int = 5, random_state: int = 42,
                                         scoring: Optional[Dict[str, str]] = None,
                                         n_jobs: int = -1) -> pd.DataFrame:
    """Ejecutar validación cruzada para varios modelos de clasificación y
    devolver un DataFrame con medias y desviaciones estándar de las métricas.

    `scoring`: diccionario nombre->scorer aceptado por scikit-learn. Si es None,
    se usan métricas comunes por defecto.
    """
    if scoring is None:
        scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, estimator in models.items():
        try:
            est = _wrap_in_pipeline_if_needed(estimator)
            res = cross_validate(est, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=n_jobs)
            row = {f"{metric}_mean": float(np.mean(values)) for metric, values in res.items() if metric.endswith("test_score")}
            row.update({f"{metric}_std": float(np.std(values)) for metric, values in res.items() if metric.endswith("test_score")})
            # normalizar claves a los nombres de scoring
            # cross_validate devuelve claves como 'test_accuracy'
            cleaned = {}
            for k, v in row.items():
                cleaned[k.replace('test_', '')] = v
            cleaned["model"] = name
            rows.append(cleaned)
        except Exception as e:
            rows.append({"model": name, "error": str(e)})
    df = pd.DataFrame(rows)
    return df


# Función: `cross_validate_models_regression`
# Qué hace: ejecutar validación cruzada para modelos de regresión y devolver métricas agregadas.
def cross_validate_models_regression(models: Dict[str, Any], X, y,
                                     cv_splits: int = 5, random_state: int = 42,
                                     scoring: Optional[Iterable[str]] = None,
                                     n_jobs: int = -1) -> pd.DataFrame:
    """Ejecutar validación cruzada para modelos de regresión y devolver métricas agregadas.

    `scoring`: lista de strings de scoring (p.ej. 'neg_mean_absolute_error').
    """
    if scoring is None:
        scoring = ["neg_mean_absolute_error", "neg_mean_squared_error", "r2"]
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, estimator in models.items():
        try:
            est = _wrap_in_pipeline_if_needed(estimator)
            res = cross_validate(est, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=n_jobs)
            row = {f"{metric}_mean": float(np.mean(values)) for metric, values in res.items() if metric.startswith('test_')}
            row.update({f"{metric}_std": float(np.std(values)) for metric, values in res.items() if metric.startswith('test_')})
            row["model"] = name
            rows.append(row)
        except Exception as e:
            rows.append({"model": name, "error": str(e)})
    df = pd.DataFrame(rows)
    return df


# Función: `save_metrics`
# Qué hace: guardar un DataFrame de métricas en CSV asegurando que exista el directorio.
def save_metrics(df: pd.DataFrame, path: str) -> None:
    """Guardar DataFrame de métricas en CSV, creando directorios si es necesario."""
    resolved = _resolve_path(path)
    if resolved is None:
        raise ValueError('Path must not be None')
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    df.to_csv(resolved, index=False)


# Función: `plot_and_save_confusion_matrix`
# Qué hace: crear la matriz de confusión, mostrarla y guardarla opcionalmente.
def plot_and_save_confusion_matrix(y_true, y_pred, labels: Optional[Iterable] = None, save_path: Optional[str] = None):
    """Trazar la matriz de confusión y opcionalmente guardarla en disco."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax)
    plt.title('Matriz de confusión')
    plt.tight_layout()
    if save_path:
        resolved = _resolve_path(save_path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        fig.savefig(resolved, dpi=150)
        plt.close(fig)


# Función: `plot_and_save_roc`
# Qué hace: calcular la curva ROC y AUC mediante validación cruzada y guardar la figura.
def plot_and_save_roc(estimator, X, y, cv, save_path: Optional[str] = None):
    """Calcular curva ROC y AUC usando `cross_val_predict` y guardar la figura si se indica.

    `estimator`: estimador (ajustado o no); `cross_val_predict` ajustará en cada fold.
    `cv`: objeto de particionado de validación cruzada.
    """
    try:
        y_proba = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')[:, 1]
    except Exception:
        # intentar con decision_function si predict_proba no está disponible
        y_proba = cross_val_predict(estimator, X, y, cv=cv, method='decision_function')
    auc = roc_auc_score(y, y_proba)
    fig, ax = plt.subplots(figsize=(6,5))
    RocCurveDisplay.from_predictions(y, y_proba, ax=ax)
    plt.title(f'Curva ROC (AUC={auc:.3f})')
    plt.tight_layout()
    if save_path:
        resolved = _resolve_path(save_path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        fig.savefig(resolved, dpi=150)
        plt.close(fig)
    return auc

