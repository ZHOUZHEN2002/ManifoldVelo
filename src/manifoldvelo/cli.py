"""Command-line training and S-only inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import fit, predict, load_reference


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(prog='manifoldvelo')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('fit', help='Fit paired preprocessed U/S matrices')
    train.add_argument('--input', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--family', required=True, choices=['cycle', 'multipeak', 'open'])
    train.add_argument('--root-id', help='Known reference cell ID; otherwise infer an RNA-only candidate')
    train.add_argument('--config', help='JSON dictionary of model config overrides')
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--device', default='cpu')
    train.add_argument('--threads', type=int, default=1)
    train.add_argument('--max-steps', type=int, help='Multipeak-only iteration budget')
    query = sub.add_parser('predict', help='Predict new cells from spliced RNA alone')
    query.add_argument('--reference', required=True)
    query.add_argument('--input', required=True)
    query.add_argument('--output', required=True)
    query.add_argument('--device', default='cpu')
    query.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    arrays = read_npz(args.input)
    if args.command == 'fit':
        permitted = {'S', 'U', 'genes', 'cell_ids', 'W_0_mask', 'n_selected_genes'}
        if set(arrays)-permitted or not {'S', 'U', 'genes', 'cell_ids'} <= set(arrays):
            parser.error('Training NPZ must contain only S, U, genes, cell_ids and optional W_0_mask/n_selected_genes')
        root = None
        if args.root_id is not None:
            index = np.flatnonzero(arrays['cell_ids'].astype(str) == args.root_id)
            if len(index) != 1:
                parser.error('--root-id must identify exactly one training cell')
            root = int(index[0])
        config = json.loads(Path(args.config).read_text()) if args.config else None
        fit(**arrays, family=args.family, output=args.output, root_index=root,
            config=config, seed=args.seed, device=args.device, max_steps=args.max_steps)
    else:
        if set(arrays) != {'S', 'genes', 'cell_ids'}:
            parser.error('Query NPZ accepts exactly S, genes and cell_ids')
        target = Path(args.output)
        if target.exists():
            raise FileExistsError(target)
        reference = load_reference(args.reference, device=args.device)
        result = predict(reference, **arrays)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **result)


if __name__ == '__main__':
    main()
