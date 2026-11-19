# Estado verificable del PSet3

## Completado
- EDA tabular, split congelado y cinco folds por grupos.
- Dos features derivadas con ablation usando solo entrenamiento (no adoptadas).
- Cinco familias con defaults, baseline ingenuo, métricas train/CV, brechas y tiempos.
- GridSearchCV: 90 configuraciones y 450 ajustes, 90 corridas hijas; mejores pipelines con firma y ejemplo.
- Evaluación única de SVR y XGBoost en holdout; política pKd>=7, top20% registrada antes de abrir test.
- Baseline de negocio aleatorio, recuperación, precisión y escenario económico explícito.
- Registry: SVR champion v1 y XGBoost challenger v2; reproducción en proceso limpio.
- Exportación docs/corridas.csv; 22 pruebas aprobadas.

## Pendientes para una entrega completa
- Redactar y maquetar el memo de hasta seis páginas con resultados, fuentes y limitaciones.
- Incorporar gráficos requeridos por el enunciado (incluidos distribución, boxplot y mapa de correlación) y captura real de la comparación en MLflow. El EDA actual prioriza tablas por decisión del usuario.
- Revisar con el profesor las adaptaciones declaradas: escalado por familia frente a “misma preparación”, y no reproducir el comparador PPB-Affinity de PSet1. El LLM se justifica como no implementado en 05.
- La evidencia del mejor pipeline de cada familia está en MLflow; no se guardaron todos los estimadores de cada fold/configuración.
- Comprobación de entrega desde una copia limpia: documentar acceso a los artefactos de datos excluidos de Git y al Registry local. No confundir código publicado con disponibilidad de los datos/volúmenes.

No hay API ni despliegue: pertenecen al PSet4. No se han hecho commits ni cambios en GitHub.
