"""Conexión explícita: nunca crear un almacén MLflow accidental junto al notebook."""
import os
import mlflow

def configurar_tracking():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ValueError("Falta MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    return mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "protein-affinity"))
