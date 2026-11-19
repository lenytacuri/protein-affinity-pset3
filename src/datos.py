"""Snapshots inmutables: lectura de Snowflake o importación de Parquet exportado."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = "PROTEIN_AFFINITY.ANALYTICS.OBT_AFFINITY"

def validar_obt(df):
    required = {"OBSERVATION_ID", "PKD"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas: {required - set(df.columns)}")
    if df.empty or df.columns.duplicated().any():
        raise ValueError("OBT vacía o columnas duplicadas")
    if df.OBSERVATION_ID.isna().any() or df.OBSERVATION_ID.duplicated().any():
        raise ValueError("OBSERVATION_ID debe ser único y no nulo")

def guardar_snapshot(df, source):
    validar_obt(df)
    folder = ROOT / "data" / "snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "obt.parquet"
    df.sort_values("OBSERVATION_ID").reset_index(drop=True).to_parquet(path, index=False)
    manifest = {"source": source, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(df), "columns": len(df.columns),
                "dtypes": {k: str(v) for k, v in df.dtypes.items()}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder

def cargar_snapshot(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "obt.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("El snapshot fue modificado")
    df = pd.read_parquet(path)
    validar_obt(df)
    if [len(df), len(df.columns)] != [manifest["rows"], manifest["columns"]]:
        raise ValueError("Dimensiones distintas del manifiesto")
    return df, manifest

def extraer_snowflake():
    import snowflake.connector
    names = ["ACCOUNT", "USER", "ROLE", "WAREHOUSE", "PRIVATE_KEY_FILE"]
    missing = [x for x in names if not os.getenv("SNOWFLAKE_" + x)]
    if missing:
        raise ValueError("Configura SNOWFLAKE_: " + ", ".join(missing))
    opts = {x.lower(): os.environ["SNOWFLAKE_" + x] for x in names}
    with snowflake.connector.connect(**opts, authenticator="SNOWFLAKE_JWT",
         session_parameters={"QUERY_TAG": "protein-affinity-pset3-snapshot", "STATEMENT_TIMEOUT_IN_SECONDS": 300}) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {TABLE} ORDER BY OBSERVATION_ID")
            query_id = cur.sfqid
            df = cur.fetch_pandas_all()
    return guardar_snapshot(df, {"table": TABLE, "query_id": query_id})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path)
    args = parser.parse_args()
    print(guardar_snapshot(pd.read_parquet(args.parquet), {"local_parquet": str(args.parquet)})
          if args.parquet else extraer_snowflake())
