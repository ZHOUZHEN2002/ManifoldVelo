# Relationship to the paper implementation

The training engines are taken from the frozen
`TSvelo-Manifold-v1.0.0-frozen.20260916` source used by the manuscript. Import
paths were moved into `manifoldvelo._core`. Dataset-launcher functions were
replaced with portable entry points; model equations, losses and optimizer
schedules were retained. Source hashes and helper extraction records are in
`provenance/SOURCE_MANIFEST.json`.

| Paper component | Released implementation |
| --- | --- |
| WT2/U2OS periodic GraphSAGE/Fourier model | `_core/cycle.py`, `_core/cycle_base/` |
| Multipeak periodic decoder and monotone clock | `_core/topo.py`, `_core/temporal.py` |
| Shared-origin open lineage curves and alternating time projection | `_core/branch.py`, `_core/branch_base/` |
| U/S-only candidate root selection in simulations | `root.py`, `_core/source_selection.py` |
| S-only reference prediction | Each trainer's `predict`, wrapped by `manifoldvelo.predict` |

The default `cycle` settings match the recorded `cyc_fourier2_pca` setup: two
Fourier harmonics, PCA initialization, 300 warm-up and 2,400 joint updates.
`multipeak` retains the recorded 18,000-update ceiling and training-only stability
checks; its kinetic fitting starts after 1,500 reconstruction updates.

`configs/simulation_paper.json` preserves the six recorded simulation parameter
sets. It is a provenance/configuration record, not a claim that a short example
reproduces the paper's scores. Use each entry's `config` as the Python `config`
argument for cycle/open models, removing `seed` or matching the explicit seed.
Pass `chart_parameters` as `chart_options` for an open model. The multipeak
backend uses its fixed paper configuration directly.

Exact reproduction of a reported experiment also needs that experiment's prepared
RNA values, gene panel, reference/query split, root protocol, graph environment
and evaluation code. This public code package supplies the fitting and inference
implementation; it does not redistribute the private experimental workspace or
claim that demonstration outputs are biological benchmark results.

The root selector's group bootstrap is a stability diagnostic. Real-data roots
and topology families in the primary manuscript comparisons were specified in
the recorded experiment protocol. Automatic branch chart construction infers
paths within the chosen open family; it does not select between cyclic and open
biology.
