"""Expression-graph charts. Cell type, fitted TSvelo time and UMAP are absent."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import networkx as nx


@dataclass
class BranchCharts:
    initial_time: np.ndarray
    membership: np.ndarray
    paths: list
    root_index: int
    diagnostics: dict

    def validate(self, n):
        t = np.asarray(self.initial_time)
        m = np.asarray(self.membership)
        if t.shape != (n,) or not np.isfinite(t).all() or t.min() < 0 or t.max() > 1:
            raise ValueError('initial_time must contain finite graph-derived values in [0,1]')
        if m.ndim != 2 or m.shape[0] != n or not np.isin(m, [0, 1]).all():
            raise ValueError('membership must be a cells x charts binary matrix')
        if np.any(m.sum(1) == 0) or np.any(m.sum(0) < 5):
            raise ValueError('Each cell needs a chart; each chart needs >=5 cells')
        if not 0 <= self.root_index < n or t[self.root_index] != 0:
            raise ValueError('The selected root must have initial time zero')
        if not np.all(m[self.root_index]):
            raise ValueError('Every root-to-terminal chart must contain the root')
        return self


def charts_from_anndata(adata, root_id, neighbors=30, resolution=.1, seed=0,
                        reuse_expression_graph=False, paga_threshold=None):
    """Build rooted paths using DPT and an expression-derived PAGA graph.

    If no threshold is supplied, a maximum spanning tree of positive PAGA
    edges is used. An explicit threshold uses unweighted shortest paths,
    matching the recorded native full-cell protocol. Disconnected cells are
    rejected instead of receiving fabricated times or silently disappearing.
    """
    import scanpy as sc
    if root_id not in adata.obs_names:
        raise ValueError('Root ID not present in training data')
    if not reuse_expression_graph:
        sc.pp.neighbors(adata, n_neighbors=min(neighbors, len(adata)-1), use_rep='X', random_state=seed)
        sc.tl.diffmap(adata, n_comps=min(15, len(adata)-2), random_state=seed)
        sc.tl.leiden(adata, resolution=resolution, random_state=seed)
        sc.tl.paga(adata, groups='leiden')
    for key in ['connectivities']:
        if key not in adata.obsp:
            raise ValueError('Missing expression graph: ' + key)
    if connected_components(adata.obsp['connectivities'], directed=False)[0] != 1:
        raise ValueError('Expression graph is disconnected; fit components separately')
    root = int(adata.obs_names.get_loc(root_id))
    adata.uns['iroot'] = root
    sc.tl.dpt(adata, n_dcs=min(10, adata.obsm['X_diffmap'].shape[1]))
    dpt = adata.obs.dpt_pseudotime.to_numpy()
    if not np.isfinite(dpt).all():
        raise ValueError('DPT has unreachable cells')
    # Same rank convention as native TSvelo; no fitted native t is read.
    rank = np.zeros(len(dpt), dtype=np.float64)
    rank[np.argsort(dpt)] = np.arange(len(dpt))
    t0 = rank / (len(rank)-1)
    if t0[root] != 0:
        # Resolve indistinguishable zero-distance roots explicitly, without
        # using any time annotations to order the remaining tied cells.
        order = np.r_[root, np.argsort(dpt)[np.argsort(dpt) != root]]
        t0[order] = np.arange(len(dpt))/(len(dpt)-1)
    labels = adata.obs.leiden.astype(str).to_numpy()
    groups = list(adata.obs.leiden.cat.categories.astype(str))
    a = adata.uns['paga']['connectivities'].copy().toarray()
    if len(groups) == 1:
        paths = [[groups[0]]]
        retained = []
    else:
        if paga_threshold is not None:
            a[a < paga_threshold] = 0
        graph = nx.from_numpy_array(a)
        if not nx.is_connected(graph):
            raise ValueError('PAGA graph does not cover all training cells at this threshold')
        if paga_threshold is None:
            graph = nx.maximum_spanning_tree(graph, weight='weight')
        start = groups.index(labels[root])
        routes = list(nx.single_source_shortest_path(graph, start).values())
        routes = [p for p in routes if not any(len(p) < len(q) and set(p).issubset(q) for q in routes)]
        paths = [[groups[j] for j in p] for p in routes]
        retained = [[groups[u], groups[v], float(d['weight'])] for u, v, d in graph.edges(data=True)]
    membership = np.column_stack([np.isin(labels, path) for path in paths])
    diagnostic = dict(source='training expression graph, root, DPT and PAGA only',
                      topology='rooted open lineage charts', paths=paths, graph_edges=retained,
                      graph_threshold=paga_threshold, cells=len(t0), charts=len(paths),
                      root_id=str(root_id), root_index=root, labels_used=False,
                      native_fitted_time_used=False, umap_used=False,
                      covered_cells=int(membership.any(1).sum()),
                      overlap_cells=int((membership.sum(1) > 1).sum()))
    return BranchCharts(t0.astype(np.float32), membership, paths, root, diagnostic).validate(len(t0))
