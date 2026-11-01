"""Train-only preprocessing and directed reference-to-query neighborhoods.

No label, U-query, batch ID, or query-query edge enters this interface.
Only prepared nonnegative S measurements are accepted; library-depth offsets
are deliberately not inferred from an unspecified gene panel.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def matrix(value,name):
    value=value.toarray() if sparse.issparse(value) else np.asarray(value)
    if value.ndim!=2 or value.shape[0]==0 or value.shape[1]==0:
        raise ValueError(name+' must be a nonempty cells x genes matrix')
    if not np.isfinite(value).all() or (value<0).any():
        raise ValueError(name+' must contain finite, nonnegative, unlogged measurements')
    return value.astype(np.float32)


def names(values,n,label):
    values=np.asarray(values,str)
    if values.shape!=(n,) or len(set(values))!=n or np.any(values==''):
        raise ValueError(label+' must contain one unique nonempty identifier per entry')
    return values


@dataclass
class STransform:
    genes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    clip: float
    pca_mean: np.ndarray
    pca_components: np.ndarray

    def apply(self,S,genes):
        S=matrix(S,'S');genes=names(genes,S.shape[1],'genes')
        if set(genes)!=set(self.genes):raise ValueError('Query gene coverage must match training genes exactly')
        lookup={g:j for j,g in enumerate(genes)}
        S=S[:,[lookup[g] for g in self.genes]]
        X=np.clip((np.log1p(S)-self.mean)/self.scale,-self.clip,self.clip).astype(np.float32)
        return X,(X-self.pca_mean)@self.pca_components.T


def fit_transform(S,genes,cfg):
    S=matrix(S,'training S');genes=names(genes,S.shape[1],'genes')
    if len(S)<5:raise ValueError('At least five training cells required')
    log=np.log1p(S);mean=log.mean(0);scale=np.maximum(log.std(0),cfg.feature_scale_floor)
    if not np.any(log.std(0)>1e-6):raise ValueError('Training expression has no variable features')
    X=np.clip((log-mean)/scale,-cfg.feature_clip,cfg.feature_clip).astype(np.float32)
    dims=min(cfg.graph_pcs,len(S)-1,S.shape[1])
    pca=PCA(dims,svd_solver='full').fit(X)
    t=STransform(genes,mean,scale,cfg.feature_clip,pca.mean_,pca.components_)
    return t,*t.apply(S,genes)


@dataclass
class ReferenceGraph:
    positions: np.ndarray
    neighbors: int
    edge_index: np.ndarray
    initial_z: np.ndarray
    diagnostics: dict

    def query_edges(self,query_positions):
        query_positions=np.asarray(query_positions)
        if query_positions.ndim!=2 or query_positions.shape[1]!=self.positions.shape[1]:
            raise ValueError('Query graph coordinates have wrong shape')
        nn=NearestNeighbors(n_neighbors=self.neighbors,n_jobs=1).fit(self.positions)
        indices=nn.kneighbors(query_positions,return_distance=False)
        # source: TRAIN node; destination: QUERY node. Never add reverse or query-query edges.
        return np.vstack((indices.ravel(),np.repeat(np.arange(len(indices)),self.neighbors))).astype(np.int64)


def build_reference_graph(positions,neighbors,seed=0):
    positions=np.asarray(positions,np.float64);n=len(positions);k=min(neighbors,n-1)
    if n<5 or not np.isfinite(positions).all():raise ValueError('Invalid training graph coordinates')
    distances,indices=NearestNeighbors(n_neighbors=k+1,n_jobs=1).fit(positions).kneighbors(positions)
    sources=[];targets=[];d=[]
    for i in range(n):
        keep=indices[i]!=i
        sources.extend(indices[i][keep][:k]);targets.extend([i]*k);d.extend(distances[i][keep][:k])
    d=np.asarray(d);positive=d[d>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    weights=np.exp(-.5*(d/max(bandwidth,1e-12))**2).clip(1e-8,1.)
    A=sparse.csr_matrix((weights,(targets,sources)),shape=(n,n))
    A=A.maximum(A.T).tocsr();A.eliminate_zeros()  # self edges were already excluded above
    components,_=csgraph.connected_components(A,directed=False)
    if components!=1:
        raise ValueError('Training graph has %d components; inspect preprocessing/batches/neighbors. '
                         'The framework does not silently join disconnected populations.'%components)
    laplacian=csgraph.laplacian(A,normed=True)
    if n<=128:
        eigval,eigvec=np.linalg.eigh(laplacian.toarray());eigval=eigval[:3];eigvec=eigvec[:,:3]
    else:
        eigval,eigvec=eigsh(laplacian,k=3,which='SM',tol=1e-7,
            v0=np.random.default_rng(seed).normal(size=n))
        order=np.argsort(eigval);eigval=eigval[order];eigvec=eigvec[:,order]
    coords=eigvec[:,1:3]/np.maximum(eigvec[:,1:3].std(0),1e-12)
    radius=np.linalg.norm(coords,axis=1)
    if (radius<1e-10).any():raise ValueError('Degenerate spectral circle initialization')
    z=(coords/radius[:,None]).astype(np.float32)
    dst,src=A.nonzero()
    info=dict(cells=n,edges=len(src),neighbors=k,connected_components=int(components),
        spectral_eigenvalues=eigval.tolist(),spectral_pair_ratio=float(eigval[2]/max(eigval[1],1e-12)),
        ring_topology_proven=False,graph_input='training log1p S only',
        spectral_graph='weighted symmetric kNN',message_passing='unweighted mean on same edges')
    return ReferenceGraph(positions.astype(np.float32),k,np.vstack((src,dst)).astype(np.int64),z,info)


def negative_edges(edge_index,n,seed,max_edges=4096):
    """Sample genuine train nonedges, excluding self; no dense NxN matrix."""
    forbidden=set((edge_index[0]*n+edge_index[1]).tolist())
    forbidden.update((np.arange(n)*(n+1)).tolist())
    if len(forbidden)==n*n:return np.empty((2,0),np.int64)
    rng=np.random.default_rng(seed);found=[]
    for _ in range(50):
        left=rng.integers(n,size=max_edges*2);right=rng.integers(n,size=max_edges*2)
        found.extend((int(a),int(b)) for a,b in zip(left,right) if int(a*n+b) not in forbidden)
        if len(found)>=max_edges:break
    if not found:raise ValueError('Unable to sample negative graph edges')
    return np.asarray(found[:max_edges],np.int64).T
