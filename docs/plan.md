# Plan aplicado al curso

Referencia local revisada: /Users/lenytacuri/data-science-end-to-end.
- Semana 08 sesión 02: notebook de exploración, código → resultado → conclusión.
- Semana 09 sesión 02: preparación importable, Pipeline y ColumnTransformer.
- Semana 11 sesión 01: MLflow, datos/semilla/commit y recuperación del modelo.
- Semana 11 sesión 02: validación por grupos y métrica de negocio.
El PSet 3 prevalece sobre ejemplos didácticos: test solo para los dos finalistas.

## Estado
Entorno preparado; sin EDA ejecutada, sin split, sin entrenamiento, sin API.
La exploración general de calidad precede al modelado. Las decisiones guiadas
por target, selección de features e imputación se toman sobre entrenamiento,
una vez congelada la separación. No eliminar atípicos por ser atípicos.

## Nuestro desafío: dependencia biológica
1. Auditar observaciones repetidas, PDB, geometrías, variantes y grupos de cadenas.
2. Definir generalización: nuevos complejos, nuevas familias o nuevos participantes.
3. Construir grupos por similitud de secuencia/estructura, considerando L/R y
   posibles inversiones. PDB diferente no prueba independencia.
4. Si el objetivo es participantes nuevos, examinar componentes conectados por
   participantes compartidos y el tamaño de los grupos resultantes.
5. Fijar método/umbral/cobertura sin optimizarlos contra el test; conservar manifiesto.
6. Hold-out por grupos y GroupKFold dentro de entrenamiento, con auditoría de solapamiento.
No se importan splits de SAbDab2: corresponden a otra colección.

## Requisitos posteriores
Tres hallazgos EDA; dos features derivadas (una motivada por EDA); baseline ingenuo
y de negocio (recuperar PSet 1); cinco familias; LLM o justificación; tres candidatos
con >=30 configuraciones y nested runs. Registro de datos/hash/commit/semilla,
train/CV media/desviación/tiempos y modelo con firma/ejemplo. Evaluación final de dos
modelos; champion/challenger; reproducción en notebook limpio; corridas.csv y memo <=6 páginas.
Kestra será opcional para ejecutar los mismos módulos, sin sustituir notebooks.

## Futuro PSet 4
Persistir preprocesamiento + estimador; cargar versión fija; predicción independiente
de Jupyter. Definir esquema de entrada y si recibimos features o estructuras.
No implementar FastAPI/Uvicorn todavía.
