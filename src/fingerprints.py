"""Fingerprints descriptivos de TODOS los mínimos Cα por participante."""
import hashlib
import json
import numpy as np
import pandas as pd

VERSION = "sorted_partner_minima_v1"

def fingerprint_minimos(ligando, receptor, decimales=None):
    """Invariante a permutar residuos y a intercambiar participantes.

    Redondear define igualdad de valores cuantizados, NO una tolerancia
    geométrica ni equivalencia estructural. No intervienen afinidad o mutaciones.
    """
    partes = []
    for valores in (ligando, receptor):
        a = np.array(valores, dtype=float, copy=True)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Se requieren mínimos finitos no negativos de ambos participantes")
        if decimales is not None:
            a = np.round(a, decimales)
        a[a == 0] = 0.0
        partes.append(tuple(sorted(a.tolist())))
    partes.sort()  # L/R intercambiables; conserva sus tamaños y multiplicidades.
    texto = json.dumps([VERSION, decimales, partes], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(texto.encode()).hexdigest()

def crear_fingerprints(minimos):
    if minimos.GEOMETRY_ID.isna().any() or minimos.GEOMETRY_ID.duplicated().any():
        raise ValueError("Debe haber una fila por GEOMETRY_ID")
    filas = []
    for fila in minimos.itertuples(index=False):
        filas.append({"GEOMETRY_ID": fila.GEOMETRY_ID,
                      "FP_N_LIGAND": len(fila.DISTANCES_L),
                      "FP_N_RECEPTOR": len(fila.DISTANCES_R),
                      "FP_MINIMA_EXACT": fingerprint_minimos(fila.DISTANCES_L, fila.DISTANCES_R),
                      "FP_MINIMA_ROUND3": fingerprint_minimos(fila.DISTANCES_L, fila.DISTANCES_R, 3)})
    return pd.DataFrame(filas)

def unir_fingerprints(datos, fingerprints):
    nuevas = [c for c in fingerprints if c != "GEOMETRY_ID"]
    base = datos.drop(columns=nuevas + ["FP_STATUS"], errors="ignore")
    resultado = base.merge(fingerprints, on="GEOMETRY_ID", how="left", validate="many_to_one", sort=False)
    resultado.index = datos.index
    if not resultado.OBSERVATION_ID.equals(datos.OBSERVATION_ID):
        raise AssertionError("Cambió el orden o número de observaciones")
    resultado["FP_STATUS"] = np.where(resultado.FP_MINIMA_EXACT.notna(), "available", "unavailable")
    return resultado
