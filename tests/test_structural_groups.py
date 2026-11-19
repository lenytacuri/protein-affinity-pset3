import numpy as np
from src.structural_groups import embedding_idist, agrupar_embeddings

def rows():
    return [dict(partner=p,chain_id=c,min_distance=3.,is_standard=True,amino_acid=aa,x=x,y=y,z=0.) for p,c,aa,x,y in [("L","A","A",0,0),("L","A","D",2,0),("R","B","K",0,3),("R","B","G",2,3)]]

def test_invariance_and_direct_formula():
    a=rows();v,_,_=embedding_idist(a)
    b=[dict(t,x=-t["y"]+50,y=t["x"]-7,chain_id={"A":"Z","B":"X"}[t["chain_id"]],partner={"L":"R","R":"L"}[t["partner"]]) for t in reversed(a)]
    np.testing.assert_allclose(v,embedding_idist(b)[0],atol=1e-14)
    aa="ACDEFGHIKLMNPQRSTVWY";h=np.eye(20)[[aa.index(t["amino_acid"]) for t in a]];nodes=[]
    for i,t in enumerate(a):
        intra=[];inter=[]
        for j,u in enumerate(a):
            if i==j:continue
            dist=sum((t[k]-u[k])**2 for k in ["x","y","z"])
            signal=np.exp(-dist/16)*h[j]
            (intra if t["chain_id"]==u["chain_id"] else inter).append(signal)
        nodes.append((h[i]+(np.mean(intra,axis=0)-np.mean(inter,axis=0))/2)/2)
    np.testing.assert_allclose(v,np.mean(nodes,axis=0),atol=1e-14)

def test_connected_groups_and_order():
    x=np.zeros((4,20));x[:,0]=[0,.03,.06,1.]
    g,i,j,d=agrupar_embeddings(["a","b","c","d"],x,.04)
    assert g[0]==g[1]==g[2] and g[3]!=g[0]
    reversed_g,*_=agrupar_embeddings(["d","c","b","a"],x[::-1],.04)
    assert g==reversed_g[::-1]
    assert (d<.04).all()
