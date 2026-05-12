# Proyecto: Análisis y Modelado — Cáncer de Piel (Chile)

Resumen reproducible y modular para entrega académica.

## Objetivo
- Estandarizar y documentar un pipeline reproducible sobre `dataset_chile_cancer_piel.csv`.
- Implementar y evaluar modelos supervisados (clasificación y regresión) y aplicar análisis no supervisado (PCA, t-SNE, clustering).

## Estructura del repositorio
- `notebooks/`: Notebooks exploratorios y de modelado (01–05).
- `src/`: Código reutilizable (preprocesamiento, entrenamiento, evaluación, tuning, unsupervised).
- `results/plots/`: Gráficos generados por los notebooks.
- `results/metrics/`: CSV/JSON con resúmenes de CV y resultados de tuning.
- `results/reports/`: Informes finales y notebooks ejecutados.
- `models/trained_models/`: Modelos serializados guardados por los pipelines.

## Requisitos e instalación
1. Crear y activar un entorno virtual (recomendado). Python 3.10+.
2. Instalar dependencias:


## Reproducibilidad y ejecución
- Ejecute los notebooks desde la raíz del repositorio para que las rutas relativas funcionen correctamente.
- Las utilidades en `src/` ahora resuelven rutas relativas contra la raíz del proyecto; al guardar métricas y plots se escribirán en `results/metrics/` y `results/plots/`.

Recomendación para ejecución no interactiva (desde la raíz del repo):

```powershell
# Ejecutar y actualizar un notebook en su lugar
jupyter nbconvert --to notebook --execute results/reports/executed_03_model_evaluation.ipynb --ExecutePreprocessor.timeout=600 --inplace

# O ejecutar todos los notebooks (puede tardar)
# for Windows PowerShell (ejecutar uno por uno o en script)
jupyter nbconvert --to notebook --execute notebooks/01_exploratory_analysis.ipynb --ExecutePreprocessor.timeout=600 --inplace
jupyter nbconvert --to notebook --execute notebooks/02_supervised_modeling.ipynb --ExecutePreprocessor.timeout=600 --inplace
jupyter nbconvert --to notebook --execute notebooks/03_model_evaluation.ipynb --ExecutePreprocessor.timeout=600 --inplace
```

## Notebooks clave
- `executed_01_exploratory_analysis.ipynb`: carga y análisis descriptivo.
- `executed_02_supervised_modeling.ipynb`: preprocesamiento y entrenamiento inicial.
- `executed_03_model_evaluation.ipynb`: evaluación cross-validated y guardado de métricas/plots en `results/`.
- `_executed_04_hyperparameter_optimization.ipynb`: búsqueda de hiperparámetros y exportación de `cv_results_`.
- `executed_05_final_analysis.ipynb`: resumen, integración y análisis final.

## Artefactos guardados
- Gráficos: `results/plots/*.png`
- Métricas tabuladas: `results/metrics/*.csv`
- Modelos serializados: `models/trained_models/*.joblib`

## Notas importantes
- Evite guardar artefactos dentro del entorno del notebook; las funciones en `src/model_evaluation.py` ya guardan métricas y figuras en el árbol `results/` de la raíz del proyecto cuando se pasan rutas relativas (por ejemplo `results/metrics/...`).
- Si desea cambiar la ruta de salida, pase una ruta absoluta o edite las constantes en los scripts que ejecuten las funciones de guardado.

