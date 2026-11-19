"""Adaptación NumPy de iDist one-hot a los Cα de Silver (interfaz Cα <10 Å).
Fórmula contrastada con PPIRef/comparison.py, commit
f95974290bf3c48ccb1a683aee59cbf89791b2cc, IDist.embed.
No equivale al protocolo de extracción heavy-atom de PPIRef.
"""
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

AA = "ACDEFGHIKLMNPQRSTVWY"
VERSION = "idist_onehot_ca10_silver_v1"

def embedding_idist(rows):
    # Interfaz entre PARTICIPANTES; pooling y mensajes por CADENA física.
    selected = [x for x in rows if x["min_distance"] < 10.0]
    if {x["partner"] for x in selected} != {"L", "R"}:
        raise ValueError("no_cross_partner_ca10_interface")
    if any(not x["is_standard"] or x["amino_acid"] not in AA for x in selected):
        raise ValueError("nonstandard_interface_residue_requires_review")
    xyz = np.array([[x[k] for k in ("x", "y", "z")] for x in selected], dtype=float)
    chains = np.array([x["chain_id"] for x in selected])
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite_coordinates")
    features = np.eye(len(AA))[[AA.index(x["amino_acid"]) for x in selected]]
    weights = np.exp(-cdist(xyz, xyz, metric="sqeuclidean") / 16.0)
    same = chains[:, None] == chains[None, :]
    intra = same.copy()
    np.fill_diagonal(intra, False)
    inter = ~same
    if (intra.sum(axis=1) == 0).any() or (inter.sum(axis=1) == 0).any():
        raise ValueError("empty_chain_neighborhood_requires_review")
    msg_intra = (weights * intra) @ features / intra.sum(axis=1)[:, None]
    msg_inter = (weights * inter) @ features / inter.sum(axis=1)[:, None]
    nodes = features / 2.0 + (msg_intra - msg_inter) / 4.0
    embedding = np.mean([nodes[chains == c].mean(axis=0) for c in sorted(set(chains))], axis=0)
    return embedding, len(selected), len(set(chains))

def agrupar_embeddings(ids, embeddings, threshold):
    """Componentes conexos de todas las parejas con distancia estricta < threshold."""
    if threshold <= 0 or len(set(ids)) != len(ids):
        raise ValueError("Umbral positivo e identificadores únicos requeridos")
    x = np.asarray(embeddings, dtype=float)
    if x.shape != (len(ids), 20) or not np.isfinite(x).all() or not len(ids):
        raise ValueError("Embeddings finitos de 20 dimensiones requeridos")
    d = cdist(x, x)
    i, j = np.where(np.triu(d < threshold, k=1))
    graph = coo_matrix((np.ones(len(i)*2), (np.r_[i,j], np.r_[j,i])), shape=(len(ids), len(ids))).tocsr()
    _, labels = connected_components(graph, directed=False)
    names = {}
    for label in np.unique(labels):
        members = sorted(ids[k] for k in np.flatnonzero(labels == label))
        names[label] = "SG_" + hashlib.sha256((VERSION + str(threshold) + "|".join(members)).encode()).hexdigest()
    groups = [names[label] for label in labels]
    assert all(groups[a] == groups[b] for a,b in zip(i,j))
    return groups, i, j, d[i,j]
