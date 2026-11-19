"""Exporta del cache Silver los mínimos completos de las geometrías de la OBT.
Ejecutar dentro de training con el cache montado en /silver-cache (solo lectura).
"""
import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.datos import ROOT, cargar_snapshot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/dataset.local.json").read_text())
    snapshot = ROOT / config["snapshot"]
    datos, manifest = cargar_snapshot(snapshot)
    ids = sorted(datos.loc[datos.GEOMETRY_STATUS.eq("computed"), "GEOMETRY_ID"].unique())
    filas = []
    for gid in ids:
        archivo = args.cache / gid / "residues.jsonl.gz"
        contenido = archivo.read_bytes()  # Falta de un archivo => error explícito.
        registros = [json.loads(line) for line in gzip.decompress(contenido).splitlines()]
        if any(x["geometry_id"] != gid for x in registros):
            raise ValueError(f"Geometría incorrecta: {gid}")
        if len({x["residue_index"] for x in registros}) != len(registros):
            raise ValueError(f"Residuos duplicados: {gid}")
        if any(x["partner"] not in {"L", "R"} for x in registros):
            raise ValueError(f"Participante desconocido: {gid}")
        vectors = {p: [x["min_distance"] for x in registros if x["partner"] == p] for p in ("L", "R")}
        for p, col in [("L", "N_LIGAND"), ("R", "N_RECEPTOR")]:
            esperado = datos.loc[datos.GEOMETRY_ID.eq(gid), col].dropna().unique()
            if len(esperado) != 1 or len(vectors[p]) != int(esperado[0]):
                raise ValueError(f"Cobertura incompleta de residuos: {gid} {p}")
        filas.append({"GEOMETRY_ID": gid, "DISTANCES_L": vectors["L"], "DISTANCES_R": vectors["R"],
                      "SOURCE_SHA256": hashlib.sha256(contenido).hexdigest()})
    destino = ROOT / "data/fingerprints" / snapshot.name
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / "minimos_completos.parquet"
    pd.DataFrame(filas).to_parquet(archivo, index=False)
    detalle = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "snapshot_sha256": manifest["sha256"], "source": "Silver cache / GEOMETRY_ID / residues.jsonl.gz",
               "selection": "all usable CA residues; no SELECTED_Q1 filter",
               "geometry_count": len(filas),
               "residue_count": sum(len(x["DISTANCES_L"])+len(x["DISTANCES_R"]) for x in filas),
               "sha256": hashlib.sha256(archivo.read_bytes()).hexdigest()}
    (destino / "manifest.json").write_text(json.dumps(detalle, indent=2))
    print(json.dumps(detalle, indent=2))

if __name__ == "__main__":
    main()
