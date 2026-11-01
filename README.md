# ManifoldVelo

**Coupling cellular trajectories with RNA kinetics on continuous expression manifolds.**

ManifoldVelo jointly estimates pseudotime, unspliced/spliced expression trajectories
and RNA velocity. It supports periodic populations, branching lineages and
unbranched progression. A fitted reference can predict time, unspliced expression
and velocity for additional cells measured with spliced RNA alone.

**License: Academic Research Only (non-commercial).** See [LICENSE](LICENSE).

[中文说明](README_CN.md) · [Input and output](docs/input_output.md) ·
[Paper implementation](docs/paper_implementation.md) · [Source provenance](provenance/SOURCE_MANIFEST.json)

## Installation

Use Python 3.10 or later in a separate environment. Install a PyTorch build
appropriate for your CPU or CUDA installation, then:

```bash
git clone https://github.com/ZHOUZHEN2002/ManifoldVelo.git
cd ManifoldVelo
python -m pip install -e .
```

For automatic open-lineage chart construction with Scanpy DPT/PAGA:

```bash
python -m pip install -e '.[graph]'
```

No GPU is required. Pass `--device cuda:0` to use an available CUDA device.

## Quick start

The demonstration generates its own small RNA matrices. It downloads no data.

```bash
python examples/make_demo.py --output demo_data
manifoldvelo fit --input demo_data/train.npz --family cycle \
  --config configs/cycle_smoke.json --output outputs/cycle_demo
manifoldvelo predict --reference outputs/cycle_demo \
  --input demo_data/query.npz --output outputs/query_prediction.npz
```

The short configuration checks the workflow; it is not a benchmark accuracy
configuration. Omitting `--config` uses the full recorded WT2/U2OS cycle schedule.
Use a new output directory for each fit. Supply `--root-id CELL_ID` to provide a
known anchor; otherwise the training U/S matrices determine a root candidate.

## Python interface

```python
import numpy as np
import manifoldvelo as mv

with np.load('prepared_train.npz', allow_pickle=False) as f:
    train = {key: f[key] for key in f.files}

reference = mv.fit(**train, family='cycle', root_index=0,
                   output='outputs/reference', device='cpu')

with np.load('prepared_query.npz', allow_pickle=False) as f:
    query = {key: f[key] for key in f.files}  # S, genes, cell_ids only
result = mv.predict(reference, **query)
np.savez_compressed('query_prediction.npz', **result)

restored = mv.load_reference('outputs/reference', device='cpu')
```

Select the trajectory family explicitly:

| Family | Model | Typical use |
| --- | --- | --- |
| `cycle` | GraphSAGE phase encoder and Fourier expression decoder | WT2/U2OS periodic populations |
| `multipeak` | Periodic neural decoder and monotone circular coordinate map | Complex multipeak cycles |
| `open` | One or more curves with a shared decoded origin | Branching or unbranched development |

For `family='open'`, the expression graph supplies the lineage charts unless a
`BranchCharts` object is provided. Chart construction requires the `graph` extra.
The family is not automatically inferred. The optional root selector identifies
an RNA-derived candidate; a cyclic anchor does not by itself identify biological
M-to-G1 phase. The paper's real-data analyses use their recorded known roots.

## RNA inputs and predictions

Training requires aligned, finite, nonnegative, **unlogged**, preprocessed S and U
matrices in comparable units (`cells × genes`). Query input contains S only.
Gene names and cell IDs must be unique. No true time, UMAP coordinates or cell-type
labels are used by the fitting interface. See [the data contract](docs/input_output.md).

Outputs include `phase` or `pseudotime`, reconstructed S, predicted U, kinetic
velocities and expression-curve tangents. Time is normalized trajectory time, not
hours. Root identity fixes the anchor, not an absolute temporal calibration.

`W_0_mask[target, regulator]` encodes a regulatory prior: **1 forbids an edge; 0
allows it**. With no supplied mask, off-diagonal edges are allowed. Put paired
U/S genes first and any auxiliary S-only genes last, setting `n_selected_genes`.

Checkpoints contain reference expression and features. This code release contains
no real-cell data or fitted checkpoints. Open-model checkpoint loading uses the
original trusted local checkpoint format; load only models you created or trust.

## Validation

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests exercise training, native velocities, cyclic closure, shared branch origins,
reference save/load, gene alignment and S-only query isolation. They are software
checks rather than repetitions of the paper's full biological benchmarks.
The tested environment and release checks are recorded in
[provenance/VALIDATION.json](provenance/VALIDATION.json).

## Citation and attribution

Please cite the accompanying manuscript, **ManifoldVelo: coupling cellular
trajectories with RNA kinetics on continuous expression manifolds**, and this
software release. Author information is in [CITATION.cff](CITATION.cff); no journal
publication status or DOI is asserted here.

The implementation builds on [TSvelo](https://github.com/lijc0804/TSvelo).
Its original MIT notice is retained in [LICENSES/TSvelo-MIT.txt](LICENSES/TSvelo-MIT.txt).
Original ManifoldVelo contributions are released under the Academic Research Only
license. Separately licensed upstream code and dependencies retain their licenses.
