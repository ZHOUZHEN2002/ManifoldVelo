"""Prepared-array training and inductive, S-only query prediction.

No phase labels, clusters, FUCCI values, test U, or test library depths are
accepted by fit/predict. All normalization statistics belong to the reference.
"""
from dataclasses import asdict
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear

from .config import ManifoldConfig
from .graph import (ReferenceGraph, STransform, build_reference_graph, fit_transform,
                    matrix, names, negative_edges)
from .losses import objective
from .models import ManifoldModel, inverse_softplus, phase, circle


class ManifoldTrainer:
    def __init__(self, config=None, device='cpu'):
        self.config = (config or ManifoldConfig()).validate()
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select device="cpu"')
        self.model = None
        self.history = []
        self.kinetic_initialization = dict(applied=False)

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _require_fitted(self):
        if self.model is None or not hasattr(self, 'reference_cache'):
            raise RuntimeError('Call fit() or load() before prediction')

    def fit(self, S, U, genes, cell_ids, W_0_mask, n_selected_genes=None,
            observation_mask=None, initial=None, verbose=True, initial_phase=None, root_mask=None):
        """Fit only on reference cells, without biological time labels.

        S/U: finite, nonnegative, unlogged, same-unit cells x genes arrays.
        Gene order: selected genes first; optional TF-only genes last.
        TF-only U is ignored and decoded as zero (the legacy TSvelo branch).
        W_0_mask: genes x genes, 1=forbidden, 0=allowed, row=target.
        observation_mask: optional boolean cells x 2G [U,S] fit mask; it
        does not hide S from the encoder. Missing S cannot be inferred here.
        initial: optional train-derived beta, gamma, W, W_bias, phase_W.
        """
        cfg = self.config.validate()
        S, U = matrix(S, 'S'), matrix(U, 'U')
        if S.shape != U.shape:
            raise ValueError('S and U must have the same cells x genes shape')
        n, g = S.shape
        root_mask = np.zeros(n, dtype=bool) if root_mask is None else np.asarray(root_mask)
        if root_mask.dtype != bool or root_mask.shape != (n,):
            raise ValueError('Root mask must be boolean and match reference cells')
        if cfg.root_weight and not root_mask.any():
            raise ValueError('Positive root weight requires candidate roots')
        self.root_mask = self._tensor(root_mask, torch.bool)
        self.genes = names(genes, g, 'genes')
        self.cell_ids = names(cell_ids, n, 'cell_ids')
        selected = g if n_selected_genes is None else n_selected_genes
        if (not isinstance(selected, (int, np.integer)) or isinstance(selected, bool)
                or not 1 <= selected <= g):
            raise ValueError('n_selected_genes must be an integer in [1, n_genes]')
        self.n_selected = int(selected)
        W_0_mask = np.asarray(W_0_mask, np.float32)
        if W_0_mask.shape != (g, g) or not np.isin(W_0_mask, [0, 1]).all():
            raise ValueError('W_0_mask must be genes x genes with 1=forbidden and 0=allowed')
        y = np.concatenate((U, S), axis=1)
        mask = np.ones(y.shape, bool) if observation_mask is None else np.asarray(observation_mask)
        if mask.shape != y.shape or mask.dtype != bool:
            raise ValueError('observation_mask must be boolean cells x 2G in [U,S] order')
        mask = mask.copy()
        mask[:, selected:g] = False
        y[:, selected:g] = 0
        active = np.r_[np.arange(selected), np.arange(g, 2 * g)]
        if (mask[:, active].sum(0) == 0).any() or (mask.sum(1) == 0).any():
            raise ValueError('Each active feature and each training cell needs observed entries')
        count = mask.sum(0).clip(1)
        mean = (y * mask).sum(0) / count
        variance = ((y - mean)**2 * mask).sum(0) / count
        scale = np.maximum(np.sqrt(variance), cfg.state_scale_floor).astype(np.float32)
        mean = mean.astype(np.float32)
        beta = np.full(g, 2 * math.pi, np.float32)
        gamma = np.clip(beta * mean[:g] / mean[g:].clip(.01), .1, 100).astype(np.float32)
        gamma[selected:] = 2 * math.pi
        bias = .5 * (beta * mean[:g] + gamma * mean[g:])
        bias[selected:] = gamma[selected:] * mean[g + selected:]
        defaults = dict(beta=beta, gamma=gamma, W=np.zeros((g, g), np.float32),
                        W_bias=bias, phase_W=np.zeros((g, 2), np.float32))
        initial = {} if initial is None else initial
        if set(initial) - set(defaults):
            raise ValueError('Unknown kinetic initialization keys: ' + str(set(initial) - set(defaults)))
        for key, value in initial.items():
            value = np.asarray(value, np.float32)
            if value.shape != defaults[key].shape or not np.isfinite(value).all():
                raise ValueError('Invalid shape or nonfinite initialization: ' + key)
            defaults[key] = value
        if any((defaults[key] <= 1e-8).any() for key in ('beta', 'gamma')):
            raise ValueError('Initial beta and gamma must be strictly positive (>1e-8)')
        self.transform, x, positions = fit_transform(S, self.genes, cfg)
        self.graph = build_reference_graph(positions, cfg.neighbors, cfg.seed)
        # Spectral eigenvectors have arbitrary handedness. Choose it using
        # TRAIN U/S kinetics, not true phase labels: compare a harmonic S
        # tangent to beta*U-gamma*S on observed selected genes.
        z0 = self.graph.initial_z
        angle0 = np.arctan2(z0[:, 0], z0[:, 1])
        design0 = np.column_stack((np.ones(n), np.sin(angle0), np.cos(angle0),
                                   np.sin(2 * angle0), np.cos(2 * angle0)))
        coef0 = np.linalg.lstsq(design0, S[:, :selected], rcond=None)[0]
        derivative0 = 2 * math.pi * np.column_stack((np.zeros(n), np.cos(angle0), -np.sin(angle0),
                                                    2 * np.cos(2 * angle0), -2 * np.sin(2 * angle0))) @ coef0
        kinetic0 = defaults['beta'][:selected] * U[:, :selected] - defaults['gamma'][:selected] * S[:, :selected]
        valid0 = mask[:, :selected] & mask[:, g:g + selected]
        derivative0 = (derivative0 / scale[g:g + selected])[valid0]
        kinetic0 = (kinetic0 / scale[g:g + selected])[valid0]
        orientation_score = float(np.dot(derivative0, kinetic0)
                                  / max(np.linalg.norm(derivative0) * np.linalg.norm(kinetic0), 1e-12))
        if orientation_score < 0:
            self.graph.initial_z[:, 0] *= -1
        self.graph.diagnostics.update(orientation_source='training U/S and positive-rate tangent agreement',
                                      orientation_score_before_flip=orientation_score,
                                      orientation_flipped=bool(orientation_score < 0),
                                      orientation_weak=bool(abs(orientation_score) < .05))
        if initial_phase is not None:
            initial_phase = np.asarray(initial_phase, np.float32)
            if initial_phase.shape != (n,) or not np.isfinite(initial_phase).all():
                raise ValueError('Initial phases must match reference cells')
            self.graph.initial_z = circle(torch.as_tensor(initial_phase)).numpy()
            self.graph.diagnostics['caller_initial_phase'] = True
        self.graph.diagnostics['supervised_roots'] = int(root_mask.sum())
        self.x, self.y, self.mask = self._tensor(x), self._tensor(y), self._tensor(mask, torch.bool)
        self.edges = self._tensor(self.graph.edge_index, torch.long)
        init = {k: self._tensor(v) for k, v in defaults.items()}
        # Preserve the caller's torch RNG state. Training itself uses a local
        # numpy generator; this network has no dropout or batch normalization.
        rng_devices = [] if self.device.type == 'cpu' else [self.device.index or 0]
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(cfg.seed)
            self.model = ManifoldModel(g, selected, cfg, self._tensor(mean), self._tensor(scale),
                                       self._tensor(W_0_mask), init).to(self.device)
        self.initial = {k: v.copy() for k, v in defaults.items()}
        self.state_mean = mean
        self.W_0_mask = W_0_mask.copy()
        initial_z = self._tensor(self.graph.initial_z)
        # Train-only ridge initialization of the circle head. This removes
        # arbitrary near-zero heads while leaving all layers trainable.
        with torch.no_grad():
            _, _, cache = self.model.encoder.reference(self.x, self.edges)
            design = torch.cat((cache[-1].double(), self.x.new_ones(n, 1).double()), dim=1)
            ridge = torch.eye(design.shape[1], dtype=torch.float64, device=self.device) * 1e-3
            ridge[-1, -1] = 1e-8
            coef = torch.linalg.solve(design.T @ design + ridge, design.T @ initial_z.double()).float()
            self.model.encoder.head.weight.copy_(coef[:-1].T)
            self.model.encoder.head.bias.copy_(coef[-1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed)
        neg = self._tensor(negative_edges(self.graph.edge_index, n, cfg.seed,
                                         max_edges=min(4096, max(n * 4, 64))), torch.long)
        self.history = []
        self.kinetic_initialization = dict(applied=False)
        start = time.monotonic()
        total_steps = cfg.warmup_steps + cfg.joint_steps
        self.model.train()
        for step in range(total_steps):
            joint = step >= cfg.warmup_steps
            if step == cfg.warmup_steps and cfg.warmup_steps > 0 and cfg.initialize_kinetics and cfg.dynamics_weight > 0:
                self._initialize_kinetics_from_curve()
            batch = self._tensor(rng.choice(n, min(cfg.batch_size, n), replace=False), torch.long)
            # Shift a complete uniform grid; retain periodic finite differences
            # while not constraining only the same collocation locations.
            p_grid = self._tensor((np.arange(cfg.collocation_points) + rng.random()) / cfg.collocation_points)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(self.model, self.x, self.y, self.mask, self.edges, neg,
                                    initial_z, batch, p_grid, cfg, joint, self.root_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss at step %d' % step)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip,
                                                  error_if_nonfinite=True)
            optimizer.step()
            row = {k: float(v.detach()) for k, v in terms.items()}
            row.update(step=step + 1, stage='joint' if joint else 'warmup', total=float(loss.detach()),
                       gradient_norm=float(grad), seconds=time.monotonic() - start)
            self.history.append(row)
            if verbose and (step == 0 or step + 1 == total_steps or (step + 1) % 50 == 0):
                print('[%s %d/%d] total=%.5f reconstruction=%.5f dynamics=%.5f root=%.5f' %
                      (row['stage'], step + 1, total_steps, row['total'], row['reconstruction'], row['dynamics'],
                       row['root']),
                      flush=True)
        self._refresh_cache()
        return self

    def _initialize_kinetics_from_curve(self):
        """Conditional positive-rate M step after the TRAIN-only curve warmup.

        Fit beta/gamma from dS/dtau, then phase drive/intercept from dU/dtau.
        Retain the supplied W (including its mask); do not use the simulator's
        parameters, held-out cells, or time labels. This avoids jointly
        flattening a well-fitted curve to match a zero-initialized phase drive.
        """
        model, g = self.model, self.model.n_genes
        p = torch.arange(max(128, self.config.collocation_points), device=self.device)
        p = p / len(p)
        y, derivative = model.curve(p)
        with torch.no_grad():
            derivative = derivative / model.clock(p)[:, None]
        # scipy 1.10's bounded TRF QR path requires double precision.
        values = y.detach().cpu().numpy().astype(np.float64)
        dy = derivative.detach().cpu().numpy().astype(np.float64)
        beta = model.dynamics.beta.detach().cpu().numpy().copy()
        gamma = model.dynamics.gamma.detach().cpu().numpy().copy()
        fallback = 0
        for gene in range(self.n_selected):
            design = np.column_stack((values[:, gene], -values[:, g + gene]))
            fitted = lsq_linear(design, dy[:, g + gene], bounds=(1e-4, 100.))
            if fitted.success and np.isfinite(fitted.x).all():
                beta[gene], gamma[gene] = fitted.x
            else:
                fallback += 1
        target = dy[:, :g] + beta * values[:, :g]
        target[:, self.n_selected:] = (dy[:, g + self.n_selected:]
                                       + gamma[self.n_selected:] * values[:, g + self.n_selected:])
        dyn = model.dynamics
        W = (dyn.W * (1 - dyn.W_0_mask)).detach().cpu().numpy()
        alpha = np.maximum(target, 0)
        response = alpha - values[:, g:] @ W.T
        angle = 2 * math.pi * p.detach().cpu().numpy()
        design = np.column_stack((np.ones(len(p)), np.sin(angle), np.cos(angle)))
        penalty = np.diag([1e-8, self.config.kinetic_ridge, self.config.kinetic_ridge])
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ response)
        with torch.no_grad():
            dyn.beta_raw.copy_(inverse_softplus(self._tensor(beta)))
            dyn.gamma_raw.copy_(inverse_softplus(self._tensor(gamma)))
            dyn.beta_initial.copy_(self._tensor(beta))
            dyn.gamma_initial.copy_(self._tensor(gamma))
            dyn.W_bias.copy_(self._tensor(coef[0]))
            dyn.phase_W.copy_(self._tensor(coef[1:].T))
        self.kinetic_initialization = dict(applied=True, source='training-fitted decoder only',
            positive_rate_bounds=[1e-4, 100.], genes_rate_fit_fallback=fallback,
            negative_transcription_target_fraction=float(np.mean(target < 0)),
            beta_min=float(beta.min()), beta_max=float(beta.max()),
            gamma_min=float(gamma.min()), gamma_max=float(gamma.max()))

    def _refresh_cache(self):
        self.model.eval()
        with torch.no_grad():
            self.reference_z, self.reference_raw, cache = self.model.encoder.reference(self.x, self.edges)
            self.reference_cache = [h.detach() for h in cache]

    def _prediction(self, z, raw, cell_ids):
        with torch.no_grad():
            p = phase(z)
            state = self.model.decoder(z)
            rho = self.model.clock(p)
            kinetic = self.model.dynamics.clock_rhs(p, state)
        _, dy_dp = self.model.curve(p, create_graph=False)
        g = self.model.n_genes
        values = dict(phase=p, circular_latent=z, latent_radius=raw.norm(dim=-1),
                      S_reconstructed=state[:, g:], U_predicted=state[:, :g],
                      velocity_S_kinetic=kinetic[:, g:], velocity_U_kinetic=kinetic[:, :g],
                      velocity_S_curve=dy_dp[:, g:] / rho[:, None],
                      velocity_U_curve=dy_dp[:, :g] / rho[:, None],
                      clock_density=rho, angular_velocity=2 * math.pi / rho)
        result = {key: value.detach().cpu().numpy() for key, value in values.items()}
        result.update(cell_ids=np.asarray(cell_ids), genes=self.genes.copy())
        return result

    def reference_prediction(self):
        self._require_fitted()
        return self._prediction(self.reference_z, self.reference_raw, self.cell_ids)

    def predict(self, S, genes, cell_ids):
        """S-only held-out inference. Query cells must be distinct from reference.

        Predictions are invariant to query order and singleton/batch grouping
        up to floating-point roundoff. Use reference_prediction() for training.
        """
        self._require_fitted()
        x, positions = self.transform.apply(S, genes)
        ids = names(cell_ids, len(x), 'query cell_ids')
        if set(ids) & set(self.cell_ids):
            raise ValueError('Query/reference IDs overlap; use reference_prediction() for training cells')
        edges = self._tensor(self.graph.query_edges(positions), torch.long)
        with torch.no_grad():
            z, raw = self.model.encoder.query(self._tensor(x), edges, self.reference_cache)
        return self._prediction(z, raw, ids)

    def diagnostics(self, points=128, integrate=True):
        """Inspect soft dynamics consistency; no independent time truth used."""
        self._require_fitted()
        if not isinstance(points, int) or points < 16:
            raise ValueError('points must be an integer >=16')
        p = torch.linspace(0, 1, points + 1, device=self.device)
        y, dy = self.model.curve(p)
        with torch.no_grad():
            rho = self.model.clock(p)
            v = self.model.dynamics.clock_rhs(p, y)
            active = self.model.decoder.active.bool()
            scale = self.model.state_scale
            rec = self.model.decoder(self.reference_z)
            counts = self.mask.sum(0)
            err = (((rec - self.y) / scale).square() * self.mask).sum(0)
            reconstruction = (err[counts > 0] / counts[counts > 0]).mean()
            alignment = torch.nn.functional.cosine_similarity((dy[:-1] / rho[:-1, None])[:, active],
                                                               v[:-1, active], dim=-1).mean()
            dense_p = torch.arange(4096, device=self.device) / 4096.
            dense_rho = self.model.clock(dense_p)
            result = dict(
                training_reconstruction_nmse=float(reconstruction),
                dynamics_nmse=float((((dy[:-1] / rho[:-1, None] - v[:-1]) / scale)[:, active]).square().mean()),
                curve_kinetic_velocity_cosine=float(alignment),
                decoder_state_seam_max=float((y[0] - y[-1]).abs().max()),
                decoder_derivative_seam_max=float((dy[0] - dy[-1]).abs().max()),
                kinetic_velocity_seam_max=float((v[0] - v[-1]).abs().max()),
                clock_density_min=float(dense_rho.min()), clock_density_max=float(dense_rho.max()),
                clock_dense_grid_mean=float(dense_rho.mean()),
                latent_resultant_length=float(self.reference_z.mean(0).norm()),
                latent_min_radius=float(self.reference_raw.norm(dim=-1).min()),
                parameters_total=sum(t.numel() for t in self.model.parameters()),
                parameters_encoder=sum(t.numel() for t in self.model.encoder.parameters()),
                graph=self.graph.diagnostics, supervised_time_labels_used=False,
                query_inputs='S only; frozen training reference; no query-query edges',
                velocity_units='arbitrary normalized cycle clock; NOT hours',
                exact_ode_solution=False,
                kinetic_initialization=self.kinetic_initialization,
            )
        if integrate:
            scale_np = scale.detach().cpu().numpy()
            active_np = active.cpu().numpy()
            decoded = y.detach().cpu().numpy()

            def rhs(t, state):
                with torch.no_grad():
                    value = self.model.phase_rhs(self._tensor(t), self._tensor(state))
                    return value.cpu().numpy().astype(float)

            solution = solve_ivp(rhs, (0, 1), decoded[0].astype(float),
                                 t_eval=np.linspace(0, 1, points + 1), rtol=1e-5, atol=1e-6, max_step=1 / 32)
            result['ode_solver_success'] = bool(solution.success)
            result['ode_solver_message'] = solution.message
            if solution.success and np.isfinite(solution.y).all():
                rollout = solution.y.T
                result['ode_rollout_curve_nmse'] = float(np.mean((((rollout - decoded) / scale_np)[:, active_np])**2))
                result['ode_rollout_seam_nrmse'] = float(np.sqrt(np.mean((((rollout[-1] - rollout[0]) / scale_np)[active_np])**2)))
                result['ode_rollout_min_state'] = float(rollout.min())
            else:
                result['ode_rollout_curve_nmse'] = None
                result['ode_rollout_seam_nrmse'] = None
        return result

    def save(self, directory):
        """Write a NEW artifact directory. Never overwrite previous results.

        checkpoint.pt contains model tensors plus reference features; do not
        publish it if reference expression is subject to sharing restrictions.
        """
        self._require_fitted()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        def cpu(value):
            return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
        transform = {k: cpu(v) if isinstance(v, np.ndarray) and k != 'genes' else v
                     for k, v in asdict(self.transform).items()}
        transform['genes'] = self.genes.tolist()
        graph = asdict(self.graph)
        for key in ('positions', 'edge_index', 'initial_z'):
            graph[key] = cpu(graph[key])
        bundle = dict(format_version=1, config=self.config.to_dict(), genes=self.genes.tolist(),
                      cell_ids=self.cell_ids.tolist(), n_selected=self.n_selected,
                      kinetic_initialization=self.kinetic_initialization,
                      transform=transform, graph=graph, state_mean=cpu(self.state_mean),
                      state_scale=cpu(self.model.state_scale), W_0_mask=cpu(self.W_0_mask),
                      initial={k: cpu(v) for k, v in self.initial.items()},
                      x=cpu(self.x), y=cpu(self.y), mask=cpu(self.mask),
                      root_mask=cpu(self.root_mask),
                      model_state={k: cpu(v) for k, v in self.model.state_dict().items()})
        torch.save(bundle, path / 'checkpoint.pt')
        (path / 'config.json').write_text(json.dumps(self.config.to_dict(), indent=2) + '\n')
        (path / 'history.json').write_text(json.dumps(self.history, indent=2, allow_nan=False) + '\n')
        (path / 'diagnostics.json').write_text(json.dumps(self.diagnostics(), indent=2, allow_nan=False) + '\n')
        return path

    @classmethod
    def load(cls, directory, device='cpu'):
        # Tensor/primitive-only bundle; disallow arbitrary pickle execution.
        bundle = torch.load(Path(directory) / 'checkpoint.pt', map_location=device, weights_only=True)
        if bundle.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint format')
        obj = cls(ManifoldConfig(**bundle['config']), device)
        obj.genes = np.asarray(bundle['genes'])
        obj.cell_ids = np.asarray(bundle['cell_ids'])
        obj.n_selected = bundle['n_selected']
        obj.kinetic_initialization = bundle.get('kinetic_initialization', dict(applied=False))
        def numpy(value):
            return value.cpu().numpy() if torch.is_tensor(value) else value
        transform = {k: numpy(v) for k, v in bundle['transform'].items()}
        transform['genes'] = obj.genes.copy()
        obj.transform = STransform(**transform)
        obj.graph = ReferenceGraph(**{k: numpy(v) for k, v in bundle['graph'].items()})
        obj.state_mean = numpy(bundle['state_mean'])
        obj.W_0_mask = numpy(bundle['W_0_mask'])
        obj.initial = {k: numpy(v) for k, v in bundle['initial'].items()}
        obj.model = ManifoldModel(len(obj.genes), obj.n_selected, obj.config,
                                  obj._tensor(obj.state_mean), bundle['state_scale'],
                                  bundle['W_0_mask'], bundle['initial']).to(device)
        obj.model.load_state_dict(bundle['model_state'])
        obj.x, obj.y, obj.mask = bundle['x'], bundle['y'], bundle['mask']
        obj.root_mask = bundle.get('root_mask', torch.zeros(len(obj.x), dtype=torch.bool)).to(device)
        obj.edges = obj._tensor(obj.graph.edge_index, torch.long)
        obj.history = json.loads((Path(directory) / 'history.json').read_text())
        obj._refresh_cache()
        return obj
