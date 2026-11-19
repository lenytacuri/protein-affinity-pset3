# Predicción de afinidad entre proteínas

Proyecto del PSet 3 de Ciencia de Datos, desarrollado por Leny Tacuri. Estima la afinidad de unión en pKd a partir de características geométricas y químicas de complejos proteicos con estructura tridimensional disponible.

El objetivo es evaluar si estas predicciones ayudan a priorizar mediciones experimentales cuando el presupuesto es limitado. La evaluación económica es retrospectiva y usa supuestos explícitos; no demuestra ahorro real ni reemplaza la confirmación experimental.

## Datos y validación

Partimos de una OBT de 12 062 registros y 280 columnas preparada en el PSet 2. Después de la selección documentada, trabajamos con 4 447 observaciones y 173 predictores. Conservamos PKD como etiqueta y el grupo estructural como variable auxiliar, fuera de las entradas del modelo.

La separación deja 3 670 observaciones en entrenamiento y 777 en prueba. Los grupos estructurales permanecen separados entre particiones y folds. Hay 293 observaciones sin representación estructural para agrupar que se mantienen exclusivamente en entrenamiento; su similitud con test sigue siendo una limitación.

## Experimentación

- Baseline ingenuo: predicción de la media.
- Cinco candidatos: Ridge, Random Forest, XGBoost, MLP y SVR.
- Preparación dentro de pipelines: imputación por mediana y escalado donde corresponde, ajustados solo en entrenamiento de cada fold.
- Dos features derivadas evaluadas por ablation, sin incorporarlas a los finalistas por falta de mejora suficiente.
- GridSearchCV de los tres mejores: 30 combinaciones por familia y cinco folds, para 450 ajustes de validación cruzada.
- MLflow registra las corridas y conserva los mejores pipelines con firma y ejemplo de entrada.

## Resultados de test

| Modelo | RMSE | MAE | Aciertos entre 155 seleccionados |
|---|---:|---:|---:|
| SVR | 1,585 | 1,235 | 131 |
| XGBoost | 1,588 | 1,229 | 137 |

El escenario selecciona el 20 % con mayor pKd predicho y cuenta como acierto un pKd experimental ≥ 7 (Kd ≤ 100 nM). La selección aleatoria del mismo tamaño encontraría 94,36 aciertos en promedio. La precisión es 84,5 % con SVR y 88,4 % con XGBoost; la recuperación es 27,7 % y 29,0 %. Por tanto, muchos candidatos prometedores quedan fuera del cupo.

SVR figura como `champion` y XGBoost como `challenger`, según el criterio técnico fijado antes de abrir test. La pequeña diferencia de error no demuestra superioridad estadística. XGBoost obtuvo más aciertos en el escenario económico.

## Herramientas y organización

Python, pandas, NumPy, scikit-learn, XGBoost, Jupyter, Docker Compose, PostgreSQL y MLflow. Snowflake corresponde al origen de la OBT; los experimentos usan un snapshot local verificado.

Los cuadernos 01–06 documentan EDA, features, candidatos, tuning, evaluación final y gráficos. La lógica reutilizable está en `src/`. `docs/corridas.csv` permite consultar las ejecuciones sin acceder al servidor MLflow local.

## Reproducción y alcance

Consultar [README.md](README.md) para preparar el entorno y ejecutar los cuadernos. Los datos, estructuras, credenciales y volúmenes de MLflow no se incluyen en Git; se requieren los artefactos de entrada o su reconstrucción según la documentación. Clonar el código no restaura automáticamente los modelos del Registry local.

La futura API pertenece al PSet 4. Este proyecto no incluye despliegue clínico, ensayos experimentales nuevos ni una garantía de generalización a cualquier pareja de proteínas.
