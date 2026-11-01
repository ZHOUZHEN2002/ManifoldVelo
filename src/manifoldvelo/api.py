"""Portable entry points around the frozen manuscript models."""
from pathlib import Path
import json
import numpy as np
from ._core.cycle import CycleConfig, CycleTrainer, run_cycle
from ._core.branch import BranchConfig, BranchTrainer
from ._core.topo import TopoTrainer, run_topo
from ._core.cycle_base.graph import matrix, names, fit_transform
from ._core.branch_base.topology import charts_from_anndata
from ._core.common import write
from .root import infer_root


def build_charts(S, genes, cell_ids, root_index, config=None, neighbors=20, resolution=.4, seed=0):
    """Build open lineage charts from training expression using Scanpy DPT/PAGA."""
    try:
        import anndata as ad
        import pandas as pd
        import scanpy  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError as error:
        raise ImportError("Automatic open charts require: pip install 'manifoldvelo[graph]'") from error
    S = matrix(S, 'S')
    genes, cell_ids = names(genes, S.shape[1], 'genes'), names(cell_ids, len(S), 'cell_ids')
    cfg = config or BranchConfig()
    _, _, positions = fit_transform(S, genes, cfg)
    obj = ad.AnnData(X=positions, obs=pd.DataFrame(index=cell_ids))
    return charts_from_anndata(obj, str(cell_ids[int(root_index)]), neighbors=neighbors,
                               resolution=resolution, seed=seed)


def fit(S, U, genes, cell_ids, *, family, output, root_index=None,
        W_0_mask=None, n_selected_genes=None, config=None, charts=None,
        chart_options=None, seed=0, device='cpu', max_steps=None):
    """Fit paired RNA and save a reusable reference to a new output directory.

    family is 'cycle' (GraphSAGE/Fourier), 'multipeak' (monotone periodic map),
    or 'open' (one or more shared-origin curves). It is an explicit analysis
    choice. If root_index is omitted, use the RNA-only candidate selector.
    W_0_mask[target, regulator] is 1 for a forbidden edge and 0 for an allowed edge.
    """
    if family not in {'cycle', 'multipeak', 'open'}:
        raise ValueError("family must be 'cycle', 'multipeak', or 'open'")
    S, U = matrix(S, 'S'), matrix(U, 'U')
    if S.shape != U.shape:
        raise ValueError('S and U must have identical shape')
    n, g = S.shape
    genes, cell_ids = names(genes, g, 'genes'), names(cell_ids, n, 'cell_ids')
    selected = g if n_selected_genes is None else int(n_selected_genes)
    if not 1 <= selected <= g:
        raise ValueError('n_selected_genes must be between 1 and the number of genes')
    forbidden = np.eye(g, dtype=np.float32) if W_0_mask is None else np.asarray(W_0_mask, np.float32)
    if forbidden.shape != (g, g) or not np.isin(forbidden, [0, 1]).all():
        raise ValueError('W_0_mask must be a binary genes x genes matrix')
    out = Path(output)
    if out.exists():
        raise FileExistsError('Use a new output directory: '+str(out))
    if root_index is None:
        root_record = infer_root(S, U, genes, cell_ids, selected)
        root_index = root_record['root_index']
        root_source = 'RNA-only inferred candidate'
    else:
        if isinstance(root_index, bool) or int(root_index) != root_index or not 0 <= int(root_index) < n:
            raise ValueError('root_index must be a valid integer row index')
        root_index = int(root_index)
        root_record = dict(root_index=root_index, root_id=str(cell_ids[root_index]))
        root_source = 'supplied root identity'
    parameters = dict(config or {})
    if 'seed' in parameters and parameters['seed'] != seed:
        raise ValueError('Set seed through the seed argument, not a conflicting config field')
    parameters.pop('seed', None)
    if family == 'multipeak' and parameters:
        raise ValueError('The paper multipeak configuration is fixed; use max_steps for a smoke run')
    if max_steps is not None and (family != 'multipeak' or int(max_steps) < 1):
        raise ValueError('max_steps is a positive multipeak-only option')
    a = dict(S=S, U=U, genes=genes, cell_ids=cell_ids, W_0_mask=forbidden,
             n_selected_genes=selected, root_index=root_index)
    if family == 'open':
        settings = dict(anchor_weight=.2, dynamics_weight=.03, seed=seed)
        settings.update(parameters)
        cfg = BranchConfig(**settings)
        if charts is None:
            options = dict(chart_options or {})
            options.setdefault('seed', seed)
            charts = build_charts(S, genes, cell_ids, root_index, config=cfg, **options)
        if charts.root_index != root_index:
            raise ValueError('charts.root_index must match the supplied or selected root')
        charts.validate(n)
        out.mkdir(parents=True)
        model = BranchTrainer(cfg, device=device).fit(S, U, genes, cell_ids, forbidden, charts,
                n_selected_genes=selected, output=out, resume=False)
    else:
        out.mkdir(parents=True)
        job = dict(variant='cyc_fourier2_pca', seed=seed, overrides=parameters)
        if family == 'multipeak':
            if max_steps is not None:
                job['steps'] = int(max_steps)
            model = run_topo(a, job, out, device)
        else:
            model = run_cycle(a, job, out, device)
    write(out/'root_selection.json', dict(source=root_source, **root_record))
    write(out/'release.json', dict(version='1.0.0', family=family, model_directory='model',
          root_source=root_source, root_index=root_index, root_id=str(cell_ids[root_index]),
          seed=seed, checkpoint_contains_reference_expression=True))
    return model


def predict(reference, S, genes, cell_ids):
    """Infer time and velocity from S only; reference parameters stay fixed."""
    S = matrix(S, 'S')
    ids = names(cell_ids, len(S), 'query cell IDs')
    names(genes, S.shape[1], 'genes')
    if set(ids) & set(reference.cell_ids):
        raise ValueError('Query and reference IDs overlap; use reference_prediction() for fitted cells')
    return reference.predict(S, genes, ids)


def load_reference(directory, device='cpu'):
    """Load a locally created reference. Open-model checkpoints must be trusted."""
    directory = Path(directory)
    meta = json.loads((directory/'release.json').read_text())
    classes = dict(cycle=CycleTrainer, multipeak=TopoTrainer, open=BranchTrainer)
    return classes[meta['family']].load(directory/'model', device=device)
