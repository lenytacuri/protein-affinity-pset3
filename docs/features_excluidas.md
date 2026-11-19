# Propuesta de columnas fuera de X

Conjunto inspeccionado: 4447 observaciones, 253 columnas. Propuesta: 173 candidatas para X, 1 etiqueta PKD y 79 columnas auxiliares fuera de X. No se ha modificado el dataframe ni entrenado modelos.

Las 293 observaciones sin grupo estructural siguen pendientes: no tratarlas como grupos individuales al construir train/test.

## Etiqueta: guardar en y, fuera de X

Es la afinidad que queremos predecir.

- `PKD`

## Identificadores

Conservar para unir, auditar y dividir; no representan medidas geométricas.

- `OBSERVATION_ID`
- `EXPERIMENT_KEY`
- `GEOMETRY_ID`
- `PDB_ID`
- `COMPLEX_ID`
- `LIGAND_CHAINS_JSON`
- `RECEPTOR_CHAINS_JSON`

## Procedencia

Conservar fuera de X para rastrear datos.

- `SOURCE_DATASET`
- `STRUCTURE_SHA256`

## Contexto descriptivo

Nombres y anotaciones no forman parte del conjunto numérico elegido. SUBGROUP_RAW se conserva para evaluación por subgrupos.

- `LIGAND_NAME`
- `RECEPTOR_NAME`
- `LIGAND_CHAINS_RAW`
- `RECEPTOR_CHAINS_RAW`
- `SUBGROUP_RAW`
- `STRUCTURE_METHOD_RAW`

## Controles de calidad

Propuesta para la primera versión: separar cobertura, estados y conteos de mediciones de las características predictivas. Pueden revisarse como análisis de sensibilidad, pero no son equivalentes a la química o geometría objetivo.

- `KD_STATUS`
- `GEOMETRY_STATUS`
- `MUTATION_STRUCTURE_STATUS`
- `BIOLOGICAL_ASSEMBLY_VALIDATED`
- `HAS_VALID_LABEL`
- `HAS_COMPUTED_GEOMETRY`
- `HAS_Q1_BOTH_PARTNERS`
- `NEEDS_VARIANT_REVIEW`
- `TEMPERATURE_STATUS`
- `TEMPERATURE_ASSUMED`
- `AFFINITY_REFERENCE_STATUS`
- `LIGAND_CHEM_CHEMISTRY_STATUS`
- `LIGAND_CHEM_N_NONSTANDARD`
- `LIGAND_CHEM_N_UNSUPPORTED`
- `LIGAND_CHEM_PROPERTY_COVERAGE`
- `LIGAND_BACKBONE_N_MEASURED_ROWS`
- `LIGAND_BACKBONE_N_PHI_PSI_COMPLETE`
- `LIGAND_BACKBONE_N_PHI`
- `LIGAND_BACKBONE_N_PSI`
- `LIGAND_BACKBONE_N_OMEGA`
- `LIGAND_BACKBONE_N_TAU_DEG`
- `LIGAND_BACKBONE_N_N_CA_LENGTH`
- `LIGAND_BACKBONE_N_CA_C_LENGTH`
- `LIGAND_BACKBONE_N_PEPTIDE_CN_LENGTH`
- `LIGAND_BACKBONE_BACKBONE_STATUS`
- `LIGAND_BACKBONE_FRACTION_PHI_PSI_COMPLETE`
- `LIGAND_BACKBONE_PHI_COVERAGE`
- `LIGAND_BACKBONE_PSI_COVERAGE`
- `LIGAND_BACKBONE_OMEGA_COVERAGE`
- `LIGAND_BACKBONE_TAU_DEG_COVERAGE`
- `LIGAND_BACKBONE_N_CA_LENGTH_COVERAGE`
- `LIGAND_BACKBONE_CA_C_LENGTH_COVERAGE`
- `LIGAND_BACKBONE_PEPTIDE_CN_LENGTH_COVERAGE`
- `RECEPTOR_CHEM_CHEMISTRY_STATUS`
- `RECEPTOR_CHEM_N_NONSTANDARD`
- `RECEPTOR_CHEM_N_UNSUPPORTED`
- `RECEPTOR_CHEM_PROPERTY_COVERAGE`
- `RECEPTOR_BACKBONE_N_MEASURED_ROWS`
- `RECEPTOR_BACKBONE_N_PHI_PSI_COMPLETE`
- `RECEPTOR_BACKBONE_N_PHI`
- `RECEPTOR_BACKBONE_N_PSI`
- `RECEPTOR_BACKBONE_N_OMEGA`
- `RECEPTOR_BACKBONE_N_TAU_DEG`
- `RECEPTOR_BACKBONE_N_N_CA_LENGTH`
- `RECEPTOR_BACKBONE_N_CA_C_LENGTH`
- `RECEPTOR_BACKBONE_N_PEPTIDE_CN_LENGTH`
- `RECEPTOR_BACKBONE_BACKBONE_STATUS`
- `RECEPTOR_BACKBONE_FRACTION_PHI_PSI_COMPLETE`
- `RECEPTOR_BACKBONE_PHI_COVERAGE`
- `RECEPTOR_BACKBONE_PSI_COVERAGE`
- `RECEPTOR_BACKBONE_OMEGA_COVERAGE`
- `RECEPTOR_BACKBONE_TAU_DEG_COVERAGE`
- `RECEPTOR_BACKBONE_N_CA_LENGTH_COVERAGE`
- `RECEPTOR_BACKBONE_CA_C_LENGTH_COVERAGE`
- `RECEPTOR_BACKBONE_PEPTIDE_CN_LENGTH_COVERAGE`

## Fingerprints y grupos

Usar para agrupamiento o auditoría, nunca como códigos categóricos de entrada. FP_N_LIGAND/RECEPTOR repiten los tamaños disponibles en N_LIGAND/RECEPTOR.

- `FP_N_LIGAND`
- `FP_N_RECEPTOR`
- `FP_MINIMA_EXACT`
- `FP_MINIMA_ROUND3`
- `FP_STATUS`
- `IDIST_STATUS`
- `IDIST_ISSUE`
- `STRUCTURAL_GROUP_PROVISIONAL`

## Indicador constante

Es constante en la selección actual. Las futuras eliminaciones estadísticas deben ajustarse sobre entrenamiento.

- `HAS_CA_CONTACTS_10`

## Redundancias que proponemos revisar después

- `LIGAND_N_SELECTED` y `LIGAND_CHEM_N_SELECTED` frente a `N_SELECTED_Q1_LIGAND`; equivalentes del receptor. Verificar las definiciones antes de quitar copias.
- Para φ, ψ y ω, la media circular, longitud resultante y varianza circular se derivan de las medias seno/coseno. Probar una representación coherente, sin asumir que todas las transformaciones perjudican a todos los modelos.
- Recuentos y fracciones de aminoácidos tienen relación con el tamaño de la interfaz; no eliminarlos automáticamente por esa relación.

Selección por correlación, varianza e imputación: aprender reglas sobre entrenamiento (y dentro de CV), no sobre test.

## Aplicación aprobada en EDA 2.7

Se retiran del dataframe central 78 columnas, conservando PKD y
STRUCTURAL_GROUP_PROVISIONAL junto con las 173 candidatas. El dataframe central
tiene 4 447 filas y 175 columnas. Ambas tablas exportan el mismo índice
FILA_MODELO; los identificadores y controles se conservan en metadatos_modelado.
PKD y el grupo se conservarán en la tabla, pero no entrarán como predictores a X.
No se hizo el split: siguen 293 observaciones sin grupo que requieren tratamiento.
