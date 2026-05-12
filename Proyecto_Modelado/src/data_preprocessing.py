"""Archivo: src/data_preprocessing.py

Utilidades para carga y preprocesamiento de datos.

Este módulo centraliza la carga del dataset, validación básica,
codificación de variables categóricas y particionado train/test con
escalado opcional. Pensado para ser importado desde los notebooks
sin cambiar su lógica existente.
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Función: `load_data`
# Qué hace: carga un archivo CSV en un `DataFrame` y valida errores básicos.
# Propósito: punto único para leer datos desde rutas usadas en los notebooks.
def load_data(path: str) -> pd.DataFrame:
    """Cargar CSV en un DataFrame con validación básica.

    Lanza `ValueError` si no se puede leer el archivo.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error cargando el dataset desde {path}: {e}")
    return df


# Función: `encode_categoricals`
# Qué hace: aplica one-hot encoding a columnas categóricas.
# Propósito: transformar variables categóricas manteniendo compatibilidad
# con nombres de columnas para los notebooks.
def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """Codificar columnas categóricas usando one-hot encoding.

    Mantiene los nombres de columnas para compatibilidad.
    """
    try:
        return pd.get_dummies(df, drop_first=drop_first)
    except Exception as e:
        raise ValueError(f"Error en codificación de categóricas: {e}")


# Función: `create_melanoma_target`
# Qué hace: crea la columna binaria `Melanoma_target` a partir de una columna fuente.
# Propósito: preparar la etiqueta objetivo para tareas de clasificación sin
# eliminar columnas originales (compatibilidad con notebooks existentes).
def create_melanoma_target(df: pd.DataFrame, source_col: str = "Cáncer familiar 1er grado (tipo)",
                           positive_label: str = "Melanoma") -> pd.DataFrame:
    """Crear una columna binaria `Melanoma_target` desde una columna fuente.

    Compatibilidad ampliada: si la columna por defecto no existe pero el dataset
    contiene `Antecedentes personales de cáncer`, se mapeará esa columna para
    mantener compatibilidad con notebooks y reportes previos que esperan
    `Melanoma_target`.
    """
    df = df.copy()
    # Caso 1: columna fuente clásica (histórico)
    if source_col in df.columns:
        df["Melanoma_target"] = df[source_col].apply(lambda x: 1 if x == positive_label else 0)
        return df

    # Caso 2: el proyecto cambió a usar 'Antecedentes personales de cáncer'
    alt = "Antecedentes personales de cáncer"
    if alt in df.columns:
        # Mapear 'Sí'->1, 'No'->0; soportar mayúsculas/minúsculas y NaNs
        def map_ans(v):
            try:
                if str(v).strip().lower() == 'sí' or str(v).strip().lower() == 'si':
                    return 1
                return 0
            except Exception:
                return 0
        df["Melanoma_target"] = df[alt].apply(map_ans)
        return df

    raise ValueError(f"Columna esperada no encontrada: tried '{source_col}' and 'Antecedentes personales de cáncer'")


# Función: `split_and_scale_classification`
# Qué hace: divide en train/test y aplica `StandardScaler` a las características.
# Propósito: obtener matrices escaladas listas para entrenar clasificadores.
def split_and_scale_classification(X: pd.DataFrame, y: pd.Series,
                                   test_size: float = 0.2,
                                   random_state: int = 42,
                                   stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Particionar datos y aplicar `StandardScaler`, devolviendo arrays escalados.

    Devuelve: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Función: `split_and_scale_regression`
# Qué hace: divide datos de regresión en train/test y aplica escalado.
# Propósito: preparar características escaladas para modelos de regresión.
def split_and_scale_regression(X: pd.DataFrame, y: pd.Series,
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Particionar datos de regresión y aplicar `StandardScaler`.

    Devuelve arrays escalados y el escalador.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
