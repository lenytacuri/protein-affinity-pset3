"""Calcula embeddings y grupos provisionales, sin usar PKD ni hacer el split."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, gzip, hashlib, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.datos import ROOT
from src.structural_groups import VERSION, embedding_idist, agrupar_embeddings

def worker(item):
    gid, cache = item
    p = Path(cache) / gid / "residues.jsonl.gz"
    raw = p.read_bytes()
    rows = [json.loads(x) for x in gzip.decompress(raw).splitlines()]
    if any(x["geometry_id"] != gid for x in rows):
        raise ValueError("Cache con geometría incorrecta")
    result = {"GEOMETRY_ID": gid, "SOURCE_SHA256": hashlib.sha256(raw).hexdigest()}
    try:
        vector, n, chains = embedding_idist(rows)
        result.update(IDIST_STATUS="computed", IDIST_ISSUE=None, N_INTERFACE=n, N_CHAINS=chains)
        result.update({f"IDIST_{i:02}": float(v) for i,v in enumerate(vector)})
    except ValueError as error:
        result.update(IDIST_STATUS="requires_review", IDIST_ISSUE=str(error))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--cache",required=True);parser.add_argument("--workers",type=int,default=os.cpu_count() or 1)
    parser.add_argument("--reuse-embeddings", action="store_true")
    args=parser.parse_args()
    config=json.loads((ROOT/"configs/structural_groups.json").read_text())
    chosen_threshold=float(config["threshold"])
    snapshot=Path(json.loads((ROOT/"configs/dataset.local.json").read_text())["snapshot"]).name
    source=ROOT/"data/fingerprints"/snapshot/"observaciones_filtradas_limpias.parquet"
    datos=pd.read_parquet(source)
    gids=sorted(datos.GEOMETRY_ID.unique())
    out=ROOT/"data/structural_groups"/snapshot;out.mkdir(parents=True,exist_ok=True)
    if args.reuse_embeddings:
        previous=json.loads((out/"manifest.json").read_text())
        if previous["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Cambió la selección: recalcular embeddings")
        if previous["algorithm"] != VERSION:
            raise ValueError("Cambió el algoritmo: recalcular embeddings")
        if hashlib.sha256((out/"embeddings.parquet").read_bytes()).hexdigest() != previous["files_sha256"]["embeddings.parquet"]:
            raise ValueError("Embeddings modificados")
        all_emb=pd.read_parquet(out/"embeddings.parquet")
        if set(all_emb.GEOMETRY_ID) != set(gids):
            raise ValueError("Geometrías distintas: recalcular embeddings")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results=[]
            for k,result in enumerate(pool.map(worker,[(g,args.cache) for g in gids],chunksize=16),1):
                results.append(result)
                if k%500==0: print(f"Embeddings: {k}/{len(gids)}",flush=True)
        all_emb=pd.DataFrame(results).sort_values("GEOMETRY_ID").reset_index(drop=True)
        all_emb.to_parquet(out/"embeddings.parquet",index=False)
    valid=all_emb[all_emb.IDIST_STATUS.eq("computed")].reset_index(drop=True)
    ids=valid.GEOMETRY_ID.tolist();x=valid[[f"IDIST_{i:02}" for i in range(20)]].to_numpy()
    diagnostics=[]
    for threshold in sorted(set(config["sensitivity_thresholds"] + [chosen_threshold])):
        groups,i,j,dist=agrupar_embeddings(ids,x,threshold)
        sizes=pd.Series(groups).value_counts()
        diagnostics.append({"threshold":threshold,"groups":len(sizes),"largest_group":int(sizes.max()),"singletons":int(sizes.eq(1).sum()),"edges":len(i)})
        if threshold==chosen_threshold:
            mapping=pd.DataFrame({"GEOMETRY_ID":ids,"STRUCTURAL_GROUP":groups})
            edge=pd.DataFrame({"GEOMETRY_ID_A":np.array(ids)[i],"GEOMETRY_ID_B":np.array(ids)[j],"IDIST_DISTANCE":dist})
            edge.to_parquet(out/"similarity_edges.parquet",index=False)
    table=all_emb[["GEOMETRY_ID","IDIST_STATUS","IDIST_ISSUE"]].merge(mapping,on="GEOMETRY_ID",how="left",validate="one_to_one")
    table.to_parquet(out/"structural_groups.parquet",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"threshold_diagnostics.csv",index=False)
    manifest={"algorithm":VERSION,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
      "upstream_commit":"f95974290bf3c48ccb1a683aee59cbf89791b2cc","interface":"cross-partner CA min distance <10 angstrom; all such residues, not Q1",
      "threshold":chosen_threshold,"threshold_decision":config,"status":"threshold_selected_pending_structural_review","uses_affinity":False,
      "pooling":"mean of physical-chain means","node_features":"20 standard amino-acid one-hot",
      "geometry_count":len(all_emb),"computed":len(valid),"requires_review":len(all_emb)-len(valid),
      "workers":args.workers,"diagnostics":diagnostics,
      "files_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/name for name in ("similarity_edges.parquet","embeddings.parquet","threshold_diagnostics.csv","structural_groups.parquet")]}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=="__main__":main()
