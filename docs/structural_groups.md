# Grupos estructurales provisionales

## Reproducción

Después de ejecutar la sección 2.5 del EDA, desde pset_3:

```bash
docker compose run --rm --no-deps -v "$(cd ../pset_2/data/silver/cache && pwd):/silver-cache:ro" training python scripts/preparar_grupos_estructurales.py --cache /silver-cache
docker compose exec training pytest -q
```

Después ejecutar la sección 2.6 de notebooks/01_eda.ipynb.

Código: src/structural_groups.py. Entradas: selección limpia y cache Silver.
Salidas en data/structural_groups/<snapshot>/: embeddings de 20 dimensiones,
grupos, parejas similares, diagnósticos de umbrales, manifiesto con hashes y
observaciones con grupos provisionales. No se cambia Snowflake ni se entrena.

## Método y alcance

Fórmula basada en IDist.embed del commit de PPIRef
f95974290bf3c48ccb1a683aee59cbf89791b2cc:
https://github.com/anton-bushuiev/PPIRef/blob/f95974290bf3c48ccb1a683aee59cbf89791b2cc/ppiref/comparison.py

Para cada nodo: one-hot/2 + (promedio mensajes intracadena menos promedio
mensajes intercadenas)/4. Cada mensaje pondera la identidad del vecino por
exp(-distancia_CA^2/16). Excluye autoaristas. Se promedian nodos por cadena física,
luego cadenas; las letras de cadena y el orden no influyen. La selección de nodos
es CA <10 Å respecto al otro participante, no Q1 ni el protocolo de extracción
por átomos pesados de PPIRef. Los parámetros publicados para otros protocolos
no garantizan aquí equivalencia. La variante se llama idist_onehot_ca10_silver_v1.

La comparación usa distancia euclídea de embeddings y componentes conexos con
aristas estrictamente inferiores al umbral. Se inspeccionan 0.03, 0.04 y 0.05;
El usuario eligió 0.04 el 18 de noviembre de 2026; se registra en configs/structural_groups.json. La comparación es estricta (<); no se ha validado biológicamente el umbral ni se ha usado rendimiento de modelos para elegirlo.
No se consultan afinidades para crear grupos ni para escoger umbrales.

## Diagnóstico y trabajo pendiente

Se mantienen las 4 447 observaciones. 4 154 embeddings calculados; 262 casos con
residuos no estándar y 31 con vecindarios intracadena vacíos requieren revisión.
No se imputan embeddings ni se asignan singleton artificiales. Con 0.04 hay 2 106 grupos y el mayor
componente contiene 245 geometrías: requiere revisar enlaces y encadenamiento
con alineamientos estructurales independientes. No modificar umbrales buscando
un rendimiento o balance conveniente. No generar train/test hasta resolver
estos casos y definir el protocolo; luego usar grupos enteros también en CV,
y auditar pares entre todas las particiones. Conservar la revisión y el criterio
independientemente del pKd. Los grupos nunca son predictores.

Para cambiar solo el agrupamiento con los embeddings existentes y verificados:

```bash
docker compose exec training python scripts/preparar_grupos_estructurales.py --cache /silver-cache --reuse-embeddings
```

Con --reuse-embeddings no se accede al cache; se verifican hash de la selección,
versión del algoritmo, hash de embeddings e IDs antes de regenerar grupos.
