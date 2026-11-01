"""Generate small runnable inputs; these are not the manuscript benchmarks."""
import argparse
from pathlib import Path
import numpy as np
from manifoldvelo._core.cycle_base.synthetic import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='demo_data')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    a = generate(cells=240, genes=12, seed=0)
    train = a['train_mask']
    np.savez_compressed(out/'train.npz', S=a['S'][train], U=a['U'][train],
                        genes=a['genes'], cell_ids=a['cell_ids'][train],
                        W_0_mask=np.eye(12, dtype=np.float32))
    np.savez_compressed(out/'query.npz', S=a['S'][~train], genes=a['genes'], cell_ids=a['cell_ids'][~train])
    print('Known reference ID for an optional anchor:', a['cell_ids'][train][0])
    print('Created paired training input and S-only queries in', out)


if __name__ == '__main__':
    main()
