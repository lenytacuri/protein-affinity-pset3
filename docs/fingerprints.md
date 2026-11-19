# Fingerprints exploratorios de mínimos completos

Código: `src/fingerprints.py`; preparación de datos: `scripts/preparar_minimos.py`;
análisis: sección 2 de `notebooks/01_eda.ipynb`.

Desde pset_3 en la terminal del Mac, con Docker iniciado y la imagen training construida:

```bash
docker compose run --rm --no-deps -v "$(cd ../pset_2/data/silver/cache && pwd):/silver-cache:ro" training python scripts/preparar_minimos.py --cache /silver-cache
```

El cache proviene de la ejecución Silver del PSet2. Si no está presente, se debe reproducir
Silver o exportar sus residuos completos; no descargar datos diferentes silenciosamente.
Luego ejecutar el EDA desde la carga del snapshot. Se guardan, en data/fingerprints/<snapshot>/,
los mínimos completos, manifiesto, fingerprints.parquet y grupos_fingerprint.csv.
Los datos grandes están ignorados por Git; el código permite reproducirlos.

Se conserva cada mínimo de cada participante, incluso fuera de Q1, su multiplicidad y
el tamaño de los participantes. Ordenar elimina dependencia del orden de los residuos y L/R.
El hash exacto representa el vector serializado; el segundo hash usa redondeo a 3 decimales
para explorar sensibilidad. No es distancia iDist, no prueba equivalencia estructural y
no resuelve correspondencia entre mutantes y afinidades. No se actualiza Snowflake ni la OBT
inmutable; las columnas FP_* se unen solo a datos en memoria y no son predictores.

## Selección aprobada para el EDA

La sección 2.2 crea datos_filtrados, una fila por fingerprint exacto con una única
afinidad entre sus observaciones sin mutación anotada. El desempate se hace por
GEOMETRY_ID y OBSERVATION_ID. No se promedian etiquetas. Los grupos pendientes y
sus observaciones se exportan aparte. Los 135 registros sin geometría calculada
no forman parte de los grupos y permanecen en datos y en el snapshot.

Salidas: observaciones_filtradas.parquet, fingerprints_pendientes.csv,
observaciones_pendientes.parquet, seleccion_manifest.json. El manifiesto guarda
reglas, conteos, hash del snapshot y hash del archivo final. Esta selección no
prueba equivalencia estructural ni sustituye el control de similitud al dividir
train/test. La exclusión de conflictos de afinidad debe declararse al evaluar.

## Actualización aprobada: 18 de noviembre de 2026

La sección 2.3 amplía datos_filtrados a 4 447 filas: conserva las 4 345 anteriores
y añade una fila uniforme por cada uno de 102 grupos discrepantes, restringiendo
a computed y not_annotated. La semilla es 42; el orden previo es fingerprint e
OBSERVATION_ID. No se modifican ni promedian pKd. La selección inicial queda en
seleccion_sin_conflicto/. Quedan 16 fingerprints pendientes.

La auditoría de alternativas y rangos de afinidad queda separada del modelo:
auditoria_afinidades.csv y candidatas_aleatorias.parquet. El manifiesto registra
la semilla, versión de pandas y hash del archivo final. Muestrear filas no equivale
a muestrear uniformemente valores distintos de pKd; tampoco elimina sesgos
experimentales. No ajustar la semilla según resultados de entrenamiento/test.
