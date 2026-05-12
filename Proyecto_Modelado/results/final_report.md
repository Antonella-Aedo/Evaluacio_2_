# Informe Final — Análisis y Modelado (Cáncer de Piel)

## Resumen ejecutivo
Este repositorio contiene el análisis exploratorio, la implementación de modelos supervisados (clasificación y regresión) y técnicas no supervisadas (reducción de dimensión y clustering) sobre el dataset `dataset_chile_cancer_piel.csv`. Los artefactos (gráficos, métricas y modelos) se encuentran en `results/`.

## Objetivos
- Preprocesar y preparar datos clínicos y de imágenes para modelado.
- Evaluar clasificadores para detección de antecedentes personales de cáncer y modelos de regresión para tamaño máximo.
- Explorar estructura de los datos con PCA/t-SNE y técnicas de clustering.

## Resumen de métricas (Cross-Validation)

### Clasificación — CV (5 folds)

| model | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| RandomForest | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |

> Notas: los resultados anteriores provienen de ejecución CV rápida incluida en `results/metrics/classification_cv_summary.csv`.

### Regresión — CV (5 folds)

| model | neg_mean_absolute_error_mean | neg_mean_squared_error_mean | r2_mean | neg_mean_absolute_error_std | neg_mean_squared_error_std | r2_std |
|---|---:|---:|---:|---:|---:|---:|
| LinearRegression | -1.5217 | -3.1872 | -0.04525 | 0.04139 | 0.15371 | 0.03112 |
| RandomForestRegressor | -1.5633 | -3.3004 | -0.08234 | 0.04127 | 0.13476 | 0.01746 |

Los CSV originales están en `results/metrics/`:
- `results/metrics/classification_cv_summary.csv`
- `results/metrics/regression_cv_summary.csv`

## Visualizaciones clave
- PCA 2D: `results/plots/pca_2d_exec.png`

![PCA 2D](plots/pca_2d_exec.png)

- t-SNE 2D: `results/plots/tsne_2d_exec.png`

![t-SNE 2D](plots/tsne_2d_exec.png)

- KMeans (k=3) sobre PCA: `results/plots/kmeans_k3_pca_exec.png`

![KMeans k=3 PCA](plots/kmeans_k3_pca_exec.png)

- DBSCAN sobre PCA: `results/plots/dbscan_pca_exec.png`

![DBSCAN PCA](plots/dbscan_pca_exec.png)

- Agglomerative (k=3) sobre PCA: `results/plots/agglomerative_pca_exec.png`

![Agglomerative PCA](plots/agglomerative_pca_exec.png)

## Conclusiones y recomendaciones
- Los clasificadores probados (Logistic Regression y Random Forest) mostraron desempeño perfecto en las ejecuciones CV rápidas realizado en este entorno; revise la calidad de la variable objetivo y el preprocesamiento para confirmar que no haya fuga de información.
- Las métricas de regresión indican error medio absoluto ≈ 1.5 cm; se recomienda explorar features adicionales o ajustes de modelado para mejorar R².
- DBSCAN no resultó útil con parámetros por defecto; ajustar `eps` y `min_samples` o normalizar características antes de clustering puede ayudar.

## Reproducibilidad y siguientes pasos
1. Para regenerar todos los artefactos y serializar modelos, ejecutar notebooks `03` y `04` completos desde la raíz del repo.
2. Para exportar modelos entrenados a `models/trained_models/`, otorgue permiso y ejecutaré los notebooks completos (operación intensiva).
3. Si desea, puedo transformar este `final_report.md` en un PDF (requiere pandoc o nbconvert).

## Resultados de optimización: GridSearchCV vs RandomizedSearchCV
Se ejecutó una comparación controlada entre GridSearchCV y RandomizedSearchCV para RandomForest (clasificación y regresión). Los resúmenes se guardaron en:

- `results/metrics/rfc_grid_vs_rand_summary.json` (clasificación)
- `results/metrics/rfr_grid_vs_rand_summary.json` (regresión, si está disponible)

Extracto (clasificación):

```
{"grid_best_params": {"max_depth": 10, "min_samples_split": 2, "n_estimators": 50}, "grid_best_score": 0.9009998022, "grid_time": 16.1953575611, "rand_best_params": {"n_estimators": 200, "min_samples_split": 5, "max_depth": 30}, "rand_best_score": 0.8999988012, "rand_time": 7.3929021358}
```

Extracto (regresión):

```
{"grid_best_params": {"max_depth": 10, "n_estimators": 50}, "grid_best_score": -1.546500618, "grid_time": 6.0743668079, "rand_best_params": {"n_estimators": 200, "max_depth": 10}, "rand_best_score": -1.54789415, "rand_time": 12.8602828979}
```

Los modelos mejor ajustados se serializaron en `models/trained_models/` como archivos `.joblib` (ej.: `rfc_grid_best.joblib`, `rfc_rand_best.joblib`, `rfr_grid_best.joblib`, `rfr_rand_best.joblib`).

## Inclusión del aprendizaje no supervisado (integrado en `05_final_analysis.ipynb`)
El flujo de aprendizaje no supervisado (PCA, KMeans, DBSCAN, Agglomerative) fue integrado en `notebooks/05_final_analysis.ipynb` y los artefactos se encuentran en `results/`:

- Plots: `results/plots/pca_2d_integrated.png`, `results/plots/tsne_2d_exec.png`, `results/plots/kmeans_k{best}_pca_integrated.png`, `results/plots/agglomerative_pca_exec.png`.
- Métricas: `results/metrics/kmeans_integrated_metrics.csv` y `results/metrics/unsupervised_metrics.csv` (cuando esté disponible).

Esto garantiza que la sección de aprendizaje no supervisado esté dentro de la estructura oficial del proyecto (no se requieren notebooks adicionales fuera de la rúbrica).

---
Generado automáticamente a partir de `results/metrics/` y `results/plots/`.


Los notebooks marcados como ejecutados (por ejemplo executed_03_model_evaluation.ipynb) fueron generados utilizando las funciones de src (p. ej. save_metrics, plot_and_save_confusion_matrix, plot_and_save_roc). Esas utilidades resuelven rutas relativas contra la raíz del repositorio y siempre escriben los artefactos en:

Métricas: results/metrics
Gráficos: results/plots
Resultados generales: results/
