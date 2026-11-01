"""RNA-only candidate initial population selection used in the simulations."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from ._core.cycle_base.graph import matrix, names
from ._core.source_selection import geometry, group_means, kinetic_residual, source_score, rank


def infer_root(S, U, genes, cell_ids, n_selected_genes=None, seed=916, bootstraps=30):
    """Return an RNA-derived root candidate and gene-subsampling support.

    For a cycle, the candidate is a computational phase anchor, not an identified
    biological M-to-G1 transition. No time or lineage annotations are accepted.
    """
    S, U = matrix(S, 'S'), matrix(U, 'U')
    if S.shape != U.shape:
        raise ValueError('S and U must have identical shape')
    n, g = S.shape
    genes, cell_ids = names(genes, g, 'genes'), names(cell_ids, n, 'cell_ids')
    selected = g if n_selected_genes is None else int(n_selected_genes)
    if not 1 <= selected <= g or n < 32 or bootstraps < 1:
        raise ValueError('Root selection needs >=32 cells, valid paired genes and >=1 bootstrap')
    pos, neighbors, adj, graph = geometry(S)
    k = min(60, max(16, round(np.sqrt(n)/2)), n)
    labels = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                            batch_size=4096, max_iter=150).fit_predict(pos)
    counts = np.bincount(labels, minlength=k)
    row, col = adj.nonzero()
    conn = np.zeros((k, k))
    np.add.at(conn, (labels[row], labels[col]), adj.data)
    conn /= np.sqrt(np.maximum(counts[:, None]*counts[None, :], 1))
    np.fill_diagonal(conn, 0)
    s, v, gamma, valid = kinetic_residual(S[:, :selected], U[:, :selected])
    if s.shape[1] < 3:
        raise ValueError('Insufficient variable paired U/S genes for automatic root selection')
    gs, gv = group_means(s, labels, k), group_means(v, labels, k)
    score, _ = source_score(gs, gv, conn)
    eligible = counts >= max(8, int(.001*n))
    if not eligible.any():
        raise ValueError('No sufficiently populated initial group; supply a root_index')
    group = int(np.argmax(np.where(eligible, rank(score), -np.inf)))
    votes = np.zeros(k, int)
    rng = np.random.default_rng(seed)
    for _ in range(bootstraps):
        columns = rng.choice(gs.shape[1], max(3, int(np.ceil(.8*gs.shape[1]))), replace=False)
        sub, _ = source_score(gs, gv, conn, columns)
        votes[np.argmax(np.where(eligible, rank(sub), -np.inf))] += 1
    members = np.flatnonzero(labels == group)
    root = int(members[np.argmin(np.square(pos[members]-pos[members].mean(0)).sum(1))])
    return dict(root_index=root, root_id=str(cell_ids[root]), group=group,
                group_size=int(counts[group]), source_score=float(score[group]),
                bootstrap_support=float(votes[group]/bootstraps), seed=int(seed),
                bootstraps=int(bootstraps), kinetic_genes=int(valid.sum()), graph=graph,
                interpretation='RNA-derived candidate; subsampling support is not a calibrated probability')
