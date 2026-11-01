"""Tiny in-model validation, not an independent biological benchmark.

Uses the complete, coupled TSvelo RHS with sparse nonzero W, positive rates,
ReLU transcription and a two-component periodic drive. A long burn-in finds
the forced orbit. Returns continuous prepared states, NOT single-cell counts.
"""
import numpy as np
from scipy.integrate import solve_ivp


def generate(cells=240, genes=12, seed=0, noise_fraction=.015):
    if cells < 40 or genes < 4:
        raise ValueError('Synthetic example needs at least 40 cells and 4 genes')
    rng = np.random.default_rng(seed)
    beta, gamma = rng.uniform(4, 9, genes), rng.uniform(3, 8, genes)
    allowed = rng.random((genes, genes)) < .2
    np.fill_diagonal(allowed, False)
    W = rng.uniform(-.10, .10, (genes, genes)) * allowed
    bias = rng.uniform(18, 28, genes)
    offset = np.linspace(0, 2 * np.pi, genes, endpoint=False)
    amp = rng.uniform(6, 10, genes)
    phase_W = amp[:, None] * np.column_stack((np.cos(offset), np.sin(offset)))

    def rhs(p, y):
        u, s = y[:genes], y[genes:]
        z = np.array([np.sin(2 * np.pi * p), np.cos(2 * np.pi * p)])
        alpha = np.maximum(W @ s + bias + phase_W @ z, 0)
        return np.r_[alpha - beta * u, beta * u - gamma * s]

    y0 = np.r_[bias / beta, bias / gamma]
    solution = solve_ivp(rhs, (-12, 1), y0, dense_output=True,
                         rtol=1e-9, atol=1e-10, max_step=1 / 64)
    if not solution.success:
        raise RuntimeError(solution.message)
    p = rng.uniform(0, 1, cells)
    clean = solution.sol(p).T
    noise = rng.normal(size=clean.shape) * noise_fraction * clean.mean(0)
    observed = np.maximum(clean + noise, 0).astype(np.float32)
    permutation = rng.permutation(cells)
    test_indices = permutation[:max(8, cells // 5)]
    train_mask = np.ones(cells, bool)
    train_mask[test_indices] = False
    grid = np.linspace(0, 1, 257)
    grid_y = solution.sol(grid).T
    return dict(S=observed[:, genes:], U=observed[:, :genes], clean_state=clean.astype(np.float32),
                true_phase=p, train_mask=train_mask, genes=np.array(['sim_gene_%02d' % j for j in range(genes)]),
                cell_ids=np.array(['sim_cell_%04d' % i for i in range(cells)]),
                W=W, W_0_mask=(~allowed).astype(np.float32), W_bias=bias, beta=beta, gamma=gamma,
                phase_W=phase_W, grid_phase=grid, grid_state=grid_y,
                true_velocity=np.stack([rhs(t, y) for t, y in zip(p, clean)]),
                generation_state_seam=float(np.max(np.abs(grid_y[0] - grid_y[-1]))),
                noise_fraction=noise_fraction)
