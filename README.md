# PSet 3 — Predicción de afinidad entre proteínas

Este proyecto compara cinco modelos de regresión para predecir pKd a partir de características estructurales. Incluye EDA, validación por grupos, pipelines sin fuga, GridSearchCV y evaluación final con MLflow.

Estado: cinco notebooks ejecutados; 90 configuraciones de tuning; SVR registrado como `champion` y XGBoost como `challenger`. Test está congelado y sus resultados no se usan para nuevos ajustes.

## Abrir en VS Code
Abre esta carpeta (pset_3), no el repositorio del profesor. Ejecuta
**Dev Containers: Reopen in Container** desde la paleta. VS Code construye el
servicio training y levanta MLflow y PostgreSQL mediante Docker Compose.
Selecciona el kernel Python del contenedor para los notebooks.

También puedes arrancar desde la terminal de esta carpeta:
```sh
docker compose up -d --build
docker compose exec training python -m pytest -q
```
MLflow: http://localhost:5051 (el curso usa 5050). No está publicado fuera del equipo.
El backend PostgreSQL y los artefactos tienen volúmenes separados y persistentes.
`docker compose down` conserva los volúmenes; no uses `down -v` para parar normalmente.

En una copia nueva crea `.env` desde `.env.example` y reemplaza la contraseña por
un valor hexadecimal aleatorio. La configuración local creada aquí no se versiona.
Dependencias centrales fijadas a las versiones del curso, no a las últimas.

## OBT y Snowflake
Completa SNOWFLAKE_* en .env con tu cuenta. Coloca la clave privada en
`.secrets/snowflake_key.p8` (solo lectura dentro del contenedor). No subir secretos.
La consulta usa SELECT, nunca modifica la OBT ni ejecuta el pipeline del PSet 2.
```sh
docker compose exec training python -m src.datos
```
Genera `data/snapshots/<fecha>/obt.parquet` y `manifest.json`, con origen,
fecha UTC, query ID, SHA-256, tamaño y tipos. No se sobrescriben snapshots.
Se verifica el hash al leer. Para importar un Parquet exportado previamente:
```sh
docker compose exec training python -m src.datos --parquet data/incoming/obt.parquet
```
Configura la ruta del snapshot en el notebook; no seleccionar automáticamente el más reciente.

## Orden de ejecución
1. `01_eda.ipynb`: depuración documentada, fingerprints, grupos, split y EDA de entrenamiento.
2. `02_features.ipynb`: pipelines, folds y ablation de dos features derivadas.
3. `03_baseline_candidatos.ipynb`: DummyMean y cinco candidatos con defaults; train/CV, brechas y tiempos.
4. `04_tuning.ipynb`: tres GridSearchCV de 30 combinaciones; nested runs y mejores pipelines guardados.
5. `05_seleccion.ipynb`: política congelada, test, selección económica, Registry, reproducción y exportación.

Reejecutar 03/04 crea nuevas corridas y actualiza reportes. El protocolo de 05 fija URIs de modelos y rechaza cambios: para reproducir la evaluación final existente, usar los artefactos y reportes congelados; no sustituir sus modelos tras ver test. La recarga del champion se comprueba en un proceso Python limpio. En una reproducción desde cero, eliminar resultados únicamente en una copia separada y ejecutar el flujo completo con datos idénticos.

Los notebooks esperan el snapshot y artefactos locales verificados de `data/` (excluidos de Git): mínimos completos y grupos estructurales además de la OBT. Consultar `docs/origen_datos.md`, `docs/fingerprints.md` y `docs/structural_groups.md` para reconstruirlos. No basta con clonar el código sin preparar esos datos; MLflow Registry también requiere sus volúmenes o repetir entrenamiento.

## Resultados finales
- Dataset: 4 447 observaciones; 173 predictores + PKD + grupo. Train: 3 670; test: 777.
- RMSE test: SVR 1,5847; XGBoost 1,5879. Roles elegidos por CV antes de test; no se demuestra superioridad estadística.
- Negocio: top 20 % = 155 ensayos; umbral pKd ≥ 7. SVR recupera 131 positivos y XGBoost 137; azar espera 94,36. Hay 473 positivos en test, así que muchos quedan sin seleccionar.
- El ahorro es un escenario de ensayos pospuestos, igual para cualquier método con el mismo cupo, no ahorro real comprobado.
- Registry: `models:/protein-affinity-modelo@champion` (SVR v1); `@challenger` (XGBoost v2).
- Evidencia: `docs/corridas.csv`, `reports/baseline/`, `reports/features/`, `reports/tuning/`, `reports/final/`.

## Decisiones y limitaciones
`configs/evaluacion_final.json` fija el protocolo previo a test. El RMSE reemplaza al MAE principal del PSet1 para penalizar errores grandes; MAE se conserva como complementaria. No se evalúa el modelo del artículo ni un LLM: las razones y alcance se documentan en 05. La comparación inicial usa preparación apropiada por familia (escalado solo para Ridge, MLP y SVR), diferencia explícita frente a la frase “misma preparación” del enunciado.

Los 293 casos sin grupo estructural conocido están solo en entrenamiento, pero su similitud con test permanece sin medir. No afirmamos generalización a toda pareja de proteínas ni sustitución de afinidad experimental. Las features derivadas se probaron y no se incorporaron a los finalistas por falta de mejora superior a la variación de CV.

## Código reutilizable
- `src/pipeline.py`: preparación y estimadores serializables.
- `src/splitting.py`: holdout y folds por grupos con casos train-only.
- `src/experimentos.py`: comparación y tuning registrados en MLflow.
- `src/datos.py`: snapshot y validación de integridad.
- `src/tracking.py`: conexión explícita a MLflow.

La API futura cargará un pipeline del Registry; no ejecutará notebooks. No se ha implementado FastAPI en este PSet.

## Kernel de los cuadernos en VS Code
Abre pset_3 con **Dev Containers: Reopen in Container**. Dentro del notebook,
selecciona **Select Kernel → Python Environments → /usr/local/bin/python**.
Si se muestra la lista de kernels Jupyter, elige **Python — Protein Affinity (PSet 3)**.
El contenedor ya incluye Python, ipykernel y las dependencias; no necesita un venv
en el Mac. El kernel con nombre se registra al crear el Dev Container.
Para registrarlo de nuevo en un contenedor existente:
```sh
docker compose exec training python -m ipykernel install --user --name protein-affinity-pset3 --display-name "Python — Protein Affinity (PSet 3)"
```
