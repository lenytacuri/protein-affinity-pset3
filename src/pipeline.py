"""Pipelines serializables. Ningún ajuste ocurre al importar o construir."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

MODELOS = ("Ridge", "RandomForestRegressor", "XGBRegressor", "MLPRegressor", "SVR")


def log1p_no_negativo(x):
    """Solo para variables no negativas por definición; no desplaza ni recorta."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("log1p requiere columnas no negativas en este proyecto")
    return np.log1p(x)


def construir_pipeline(modelo, features, transformacion="ninguna", columnas_transformar=None,
                       seed=42, n_jobs=-1, parametros_modelo=None):
    features = list(features)
    columnas = list(columnas_transformar or [])
    if modelo not in MODELOS:
        raise ValueError(f"Modelo desconocido: {modelo}")
    if not features or len(features) != len(set(features)):
        raise ValueError("Features deben ser únicas y no vacías")
    if {"PKD", "STRUCTURAL_GROUP_PROVISIONAL"} & set(features):
        raise ValueError("Etiqueta y grupo no son predictores")
    if len(columnas) != len(set(columnas)) or not set(columnas) <= set(features):
        raise ValueError("Columnas de transformación inválidas")
    if transformacion not in {"ninguna", "log1p", "yeo-johnson"}:
        raise ValueError("Transformación desconocida")
    if (transformacion == "ninguna" and columnas) or (transformacion != "ninguna" and not columnas):
        raise ValueError("Especificar columnas únicamente cuando se solicita transformación")

    def rama(transformar=False):
        pasos = [("imputacion", SimpleImputer(strategy="median", keep_empty_features=True))]
        if transformar:
            t = (FunctionTransformer(log1p_no_negativo, feature_names_out="one-to-one")
                 if transformacion == "log1p" else PowerTransformer(method="yeo-johnson", standardize=False))
            pasos.append(("transformacion", t))
        return Pipeline(pasos)

    ramas = []
    if columnas:
        ramas.append(("transformadas", rama(True), columnas))
    resto = [c for c in features if c not in columnas]
    if resto:
        ramas.append(("originales", rama(), resto))
    preparacion = ColumnTransformer(ramas, remainder="drop", verbose_feature_names_out=False)
    modelos = {
        "Ridge": Ridge(),
        "RandomForestRegressor": RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
        "XGBRegressor": XGBRegressor(random_state=seed, n_jobs=n_jobs),
        "MLPRegressor": MLPRegressor(random_state=seed),
        "SVR": SVR(),
    }
    estimador = modelos[modelo]
    if parametros_modelo:
        estimador.set_params(**parametros_modelo)
    return Pipeline([
        ("preparacion", preparacion),
        ("escalado", StandardScaler() if modelo in {"Ridge", "MLPRegressor", "SVR"} else "passthrough"),
        ("modelo", estimador),
    ])


def agregar_features_derivadas(frame, incluir=("LOG1P_SELECTED_PAIR_COUNT", "PARTNER_SIZE_BALANCE")):
    """Features disponibles desde geometría; no dependen de afinidad ni de otros registros."""
    resultado = frame.copy()
    if "LOG1P_SELECTED_PAIR_COUNT" in incluir:
        resultado["LOG1P_SELECTED_PAIR_COUNT"] = log1p_no_negativo(resultado["SELECTED_PAIR_COUNT"])
    if "PARTNER_SIZE_BALANCE" in incluir:
        a, b = resultado["N_LIGAND"], resultado["N_RECEPTOR"]
        resultado["PARTNER_SIZE_BALANCE"] = 2 * np.minimum(a, b) / (a+b).where((a+b)>0)
    return resultado
