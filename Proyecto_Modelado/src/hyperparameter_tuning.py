"""Archivo: src/hyperparameter_tuning.py

Envolturas para ajuste de hiperparámetros con GridSearchCV y RandomizedSearchCV.

Funciones de conveniencia para ejecutar búsquedas y persistir resultados
para reproducibilidad.
"""
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Función: `grid_search_cv`
# Qué hace: ejecuta `GridSearchCV` sobre un estimador y devuelve el objeto ajustado.
def grid_search_cv(estimator, param_grid, X, y, cv=5, scoring=None, n_jobs=-1):
    """Ejecuta GridSearchCV y devuelve el objeto GridSearchCV ajustado."""
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X, y)
    return grid


# Función: `randomized_search_cv`
# Qué hace: ejecuta `RandomizedSearchCV` y devuelve el objeto ajustado.
def randomized_search_cv(estimator, param_distributions, X, y, n_iter=10, cv=5, scoring=None, n_jobs=-1, random_state=None):
    rs = RandomizedSearchCV(estimator, param_distributions, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=random_state)
    rs.fit(X, y)
    return rs


# Función: `grid_search_and_save`
# Qué hace: ejecuta GridSearchCV y guarda `cv_results_` en CSV en la ruta indicada.
def grid_search_and_save(estimator, param_grid, X, y, out_csv: str, cv=5, scoring=None, n_jobs=-1):
    """Ejecuta GridSearchCV y guarda `cv_results_` como CSV en `out_csv`.

    Devuelve el objeto GridSearchCV ajustado.
    """
    grid = grid_search_cv(estimator, param_grid, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    try:
        pd.DataFrame(grid.cv_results_).to_csv(out_csv, index=False)
    except Exception:
        # best effort: ignore saving errors
        pass
    return grid


# Función: `randomized_search_and_save`
# Qué hace: ejecuta RandomizedSearchCV y guarda `cv_results_` en CSV.
def randomized_search_and_save(estimator, param_distributions, X, y, out_csv: str, n_iter=10, cv=5, scoring=None, n_jobs=-1, random_state=None):
    rs = randomized_search_cv(estimator, param_distributions, X, y, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=random_state)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    try:
        pd.DataFrame(rs.cv_results_).to_csv(out_csv, index=False)
    except Exception:
        pass
    return rs
