# Input and output contract

## Training NPZ

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `S` | cells × genes, float | Preprocessed spliced RNA |
| `U` | cells × genes, float | Preprocessed unspliced RNA |
| `genes` | genes, Unicode strings | Unique gene identifiers |
| `cell_ids` | cells, Unicode strings | Unique reference cell identifiers |
| `W_0_mask` | genes × genes, optional | Row target / column regulator; 1 forbidden, 0 allowed |
| `n_selected_genes` | scalar integer, optional | Number of paired genes at the front of the panel |

Matrices must be finite, nonnegative and unlogged. Keep U/S in comparable units.
The model applies log1p and a reference-only feature transform internally for its
expression graph; its RNA decoder and kinetic equations operate in input units.
No biological labels belong in the training NPZ accepted by the CLI.

An already preprocessed AnnData object can be exported without changing its RNA
values. Only use the intended gene panel, in the same order in both layers:

```python
import numpy as np
from scipy import sparse

def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)

np.savez_compressed('prepared_train.npz',
    S=dense(adata.layers['spliced']).astype('float32'),
    U=dense(adata.layers['unspliced']).astype('float32'),
    genes=adata.var_names.to_numpy(dtype=str),
    cell_ids=adata.obs_names.to_numpy(dtype=str))
```

Gene filtering and RNA normalization should be chosen for the dataset before this
export. Exporting is not a replacement for that preprocessing. UMAP coordinates
are display coordinates and are not required by the model.

## Query NPZ

Contains exactly `S`, `genes`, and `cell_ids`. It must cover the reference gene
panel; gene order may differ. Query IDs must not overlap reference IDs. To obtain
fitted-cell outputs, call `reference.reference_prediction()`.

## Output

Both reference and query output include native `velocity_S_kinetic` and
`velocity_U_kinetic`, and curve derivatives `velocity_S_curve` and
`velocity_U_curve`. These are distinct estimates constrained during fitting.
`S_reconstructed` and `U_predicted` describe the decoded expression state.

The cycle models return `phase` in [0, 1). Open models return `pseudotime` in [0, 1]
and chart membership/time arrays. A cell receives a global result from its active
charts; an unused cell–chart pair has no chart-specific time. Such pairs use NaN
in the per-chart array, not in the global time or velocity output.

`release.json` records the chosen family, root source and model directory.
`root_selection.json` records the supplied anchor or RNA-only candidate support.
Reference parameters and transformations are not updated during query prediction.
