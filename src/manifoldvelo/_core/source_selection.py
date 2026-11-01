import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def normalize_columns(x):
    x = np.asarray(x, np.float64)
    return np.clip((x-x.mean(0))/np.maximum(x.std(0), .1), -5, 5)

def rank(x):
    return (rankdata(x, method='average')-.5)/len(x)

def geometry(s):
    x = normalize_columns(np.log1p(s))
    pca = PCA(n_components=min(20, x.shape[1], len(x)-1), svd_solver='full')
    pos = pca.fit_transform(x).astype(np.float32)
    distances, neighbors = NearestNeighbors(n_neighbors=min(31,len(s)), n_jobs=1).fit(pos).kneighbors(pos)
    neighbors, distances = neighbors[:,1:], distances[:,1:]
    rows = np.repeat(np.arange(len(s)), neighbors.shape[1])
    scale = max(float(np.median(distances)), 1e-8)
    weights = np.exp(-.5*(distances.ravel()/scale)**2)
    adj = sparse.csr_matrix((weights, (rows,neighbors.ravel())), shape=(len(s),len(s)))
    adj = adj.maximum(adj.T)
    components, labels = connected_components(adj)
    return pos, neighbors, adj, dict(components=int(components), component_sizes=np.bincount(labels).tolist())

def group_means(x, labels, k):
    counts = np.bincount(labels, minlength=k)
    out = np.zeros((k,x.shape[1]), np.float64)
    np.add.at(out, labels, x)
    return out/np.maximum(counts[:,None],1)

def stationary(weights, restart=.03):
    weights = np.asarray(weights, np.float64).copy()
    weights[np.diag_indices_from(weights)] += .1*weights.sum(1)+1e-10
    p = weights/weights.sum(1,keepdims=True)
    pi = np.ones(len(p))/len(p)
    for _ in range(2000):
        new = (1-restart)*(pi@p)+restart/len(pi)
        if np.max(np.abs(new-pi)) < 1e-12:
            return new
        pi = new
    return pi

def source_score(gs, gv, connectivity, columns=None):
    if columns is not None:
        gs, gv = gs[:,columns], gv[:,columns]
    delta = gs[None,:,:]-gs[:,None,:]
    delta -= delta.mean(2,keepdims=True)
    v = gv-gv.mean(1,keepdims=True)
    cosine = np.einsum('ijg,ig->ij',delta,v)
    cosine /= np.maximum(np.linalg.norm(delta,axis=2)*np.linalg.norm(v,axis=1)[:,None],1e-12)
    forward = connectivity*np.exp(4*np.clip(cosine,-1,1))
    # Initial states attract diffusion on the reversed directed graph.
    root_mass = stationary(forward.T)
    null_mass = stationary(connectivity)
    return np.log(np.maximum(root_mass,1e-12)/np.maximum(null_mass,1e-12)), cosine

def kinetic_residual(s, u):
    s, u = np.asarray(s,np.float64), np.asarray(u,np.float64)
    joint = s/np.maximum(np.max(s,0),1e-8)+u/np.maximum(np.max(u,0),1e-8)
    extreme = joint >= np.quantile(joint,.95,axis=0)
    gamma = (extreme*s*u).sum(0)/np.maximum((extreme*s*s).sum(0),1e-10)
    valid = (s.std(0)>1e-5)&(u.std(0)>1e-5)&(gamma>0)
    scale = np.maximum(s.std(0),.1)
    return s[:,valid]/scale[valid], (u[:,valid]-gamma[valid]*s[:,valid])/scale[valid], gamma, valid
