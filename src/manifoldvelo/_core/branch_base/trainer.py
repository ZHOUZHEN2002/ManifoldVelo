"""Fit open lineage charts from graph initialization and observed S/U only."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import time
import numpy as np
import torch
from scipy.optimize import lsq_linear
from sklearn.neighbors import NearestNeighbors
from ..cycle_base.graph import fit_transform, matrix, names, STransform
from ..cycle_base.models import inverse_softplus
from .config import BranchConfig
from .models import OpenChartModel
from .topology import BranchCharts


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class BranchManifoldTrainer:
    config_type = BranchConfig

    def __init__(self, config=None, device='cpu'):
        self.config = (config or self.config_type()).validate()
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')

    def tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def fit(self, S, U, genes, cell_ids, W_0_mask, charts, n_selected_genes=None,
            output=None, resume=False, verbose=True):
        cfg = self.config
        S, U = matrix(S, 'S'), matrix(U, 'U')
        if S.shape != U.shape:
            raise ValueError('S and U must have identical shape')
        n, g = S.shape
        charts.validate(n)
        self.genes = names(genes, g, 'genes')
        self.cell_ids = names(cell_ids, n, 'cell_ids')
        selected = g if n_selected_genes is None else int(n_selected_genes)
        if not 1 <= selected <= g:
            raise ValueError('Invalid number of selected genes')
        forbidden = np.asarray(W_0_mask, np.float32)
        if forbidden.shape != (g, g) or not np.isin(forbidden, [0, 1]).all():
            raise ValueError('W_0_mask must be genes x genes with 1 meaning forbidden')
        self.charts, self.n_selected = charts, selected
        self.transform, _, self.positions = fit_transform(S, self.genes, cfg)
        y = np.concatenate([U.copy(), S], -1)
        y[:, selected:g] = 0
        mean, scale = y.mean(0), np.maximum(y.std(0), cfg.state_scale_floor)
        root_neighbors = np.argsort(charts.initial_time)[:min(10, n)]
        root_state = y[root_neighbors].mean(0)
        devices = [] if self.device.type == 'cpu' else [self.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg.seed)
            self.model = OpenChartModel(charts.membership.shape[1], g, selected, cfg,
                    self.tensor(mean), self.tensor(scale), self.tensor(root_state), self.tensor(forbidden)).to(self.device)
        self.model_inputs = dict(mean=mean, scale=scale, root_state=root_state, forbidden=forbidden)
        self.y = self.tensor(y)
        self.t0 = self.tensor(charts.initial_time)
        self.membership = self.tensor(charts.membership, torch.bool)
        self.initialize_extra()
        self.history = []
        self.kinetic_initialization = []
        self.source_S, self.source_U = S, U
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        rng = np.random.default_rng(cfg.seed)
        start = 0
        out = None if output is None else Path(output)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            checkpoint = out/'continuation.pt'
            if checkpoint.exists():
                if not resume:
                    raise FileExistsError('Existing training checkpoint; pass resume=True explicitly')
                state = torch.load(checkpoint, map_location=self.device, weights_only=False)
                assert state['config'] == asdict(cfg)
                np.testing.assert_array_equal(state['cell_ids'], self.cell_ids)
                np.testing.assert_array_equal(state['genes'], self.genes)
                self.model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                rng.bit_generator.state = state['rng']
                self.history = state['history']
                self.kinetic_initialization = state['kinetic_initialization']
                self.restore_extra(state.get('extra', {}))
                start = state['step']
            protocol = dict(config=asdict(cfg), topology=charts.diagnostics,
                actual_training=True, native_fitted_times_used=False, label_times_used=False,
                output_time='branch-size weighted trained open clocks; per-branch coordinates also exported',
                stop='fixed prespecified step budget; no selection by reference agreement or truth',
                clock='positive interval lengths preserve initial within-chart order; endpoints not identified',
                curve='branch-specific open neural curves, exact common root state, soft overlap consistency',
                limitation='The clock cannot correct an incorrectly ordered expression-graph initialization')
            protocol.update(self.protocol_details())
            json_write(out/'training_protocol.json', protocol)
        total = cfg.warmup_steps+cfg.joint_steps
        started = time.monotonic()
        for step in range(start+1, total+1):
            if step == cfg.warmup_steps+1 and cfg.dynamics_weight > 0:
                self.initialize_kinetics()
            fraction = (step-1)/max(total-1, 1)
            learning_rate = cfg.final_learning_rate + .5*(cfg.learning_rate-cfg.final_learning_rate)*(1+math.cos(math.pi*fraction))
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            batch = self.tensor(rng.choice(n, min(n, cfg.batch_size), replace=False), torch.long)
            # Open interval collocation, without a last-to-first curvature term.
            grid = self.tensor((np.arange(cfg.collocation_points)+rng.random())/cfg.collocation_points)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = self.objective(batch, grid, step > cfg.warmup_steps)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective at step %d'%step)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            self.after_step(step)
            if step == 1 or step % 200 == 0 or step == total:
                with torch.no_grad():
                    clocks = self.training_clock_values(torch.arange(n, device=self.device))
                row = dict(step=step, stage='joint' if step > cfg.warmup_steps else 'warmup',
                    loss=float(loss.detach()), gradient=float(grad), learning_rate=learning_rate,
                    mean_time_change=float(torch.abs(clocks-self.t0[:, None])[self.membership].mean()),
                    seconds=time.monotonic()-started, **{k:float(v.detach()) for k,v in terms.items()})
                self.history.append(row)
                if verbose:
                    print(json.dumps(row), flush=True)
                if out is not None:
                    state = dict(model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                        step=step, rng=rng.bit_generator.state, history=self.history,
                        kinetic_initialization=self.kinetic_initialization,
                        cell_ids=self.cell_ids, genes=self.genes, config=asdict(cfg), extra=self.extra_state())
                    torch.save(state, out/'continuation.tmp')
                    (out/'continuation.tmp').replace(out/'continuation.pt')
                    json_write(out/'status.json', dict(complete=False, steps=total, updated=time.time(), **row))
                    json_write(out/'history.json', self.history)
        self.model.eval()
        self.training_steps = total
        if out is not None:
            self.save(out/'model', allow_existing=resume)
            prediction = self.reference_prediction()
            np.savez_compressed(out/'prediction.npz', **prediction)
            json_write(out/'diagnostics.json', self.diagnostics())
            json_write(out/'status.json', dict(complete=True, step=total, updated=time.time(),
                cells=n, branches=self.model.branches, stop='completed fixed training budget',
                convergence_claimed=False))
        return self

    def clock_values(self, t0):
        return torch.stack([self.model.clock(t0, torch.full_like(t0, b, dtype=torch.long))
                            for b in range(self.model.branches)], -1)

    def training_clock_values(self, batch):
        return self.clock_values(self.t0[batch])

    def state_scale(self, branch):
        return self.model.scale

    def initialize_extra(self):
        pass

    def restore_extra(self, state):
        pass

    def extra_state(self):
        return {}

    def after_step(self, step):
        pass

    def protocol_details(self):
        return {}

    def objective(self, batch, grid, joint):
        model, cfg = self.model, self.config
        active = model.active.bool()
        zero = model.root_state_raw.sum()*0
        reconstruction, overlap, dynamics, curvature = zero, zero, zero, zero
        clocks = self.training_clock_values(batch)
        count = self.membership[batch].sum(-1).clamp_min(1)
        predicted_sum = self.y.new_zeros((len(batch), self.y.shape[1]))
        predicted_square = self.y.new_zeros((len(batch), self.y.shape[1]))
        for b in range(model.branches):
            member = self.membership[batch, b]
            if member.any():
                yhat = model.decode(clocks[member, b], b)
                residual = (yhat-self.y[batch[member]])/self.state_scale(b)
                reconstruction = reconstruction + residual[:, active].square().mean()/model.branches
                normalized = yhat/model.scale
                predicted_sum[member] += normalized
                predicted_square[member] += normalized.square()
            if joint:
                curve, derivative = model.curve(grid, b, create_graph=True)
                dynamics = dynamics + ((derivative-model.rhs(grid, curve, b))/self.state_scale(b))[:, active].square().mean()/model.branches
                curvature = curvature + ((derivative[1:]-derivative[:-1])*len(grid)/self.state_scale(b))[:, active].square().mean()/model.branches
        common = count > 1
        if common.any():
            variance = predicted_square[common]/count[common, None] - (predicted_sum[common]/count[common, None]).square()
            overlap = variance[:, active].clamp_min(0).mean()
        anchor = (clocks-self.t0[batch, None]).square()[self.membership[batch]].mean()
        smooth = model.clock.penalty()
        rate = ((model.W*(1-model.forbidden)).square().mean()
                + (model.drive/model.alpha_scale[None, :, None]).square().mean()
                + .01*(torch.log(model.beta).square().mean()+torch.log(model.gamma).square().mean()))
        total = reconstruction + cfg.anchor_weight*anchor + cfg.overlap_weight*overlap + cfg.smoothness_weight*smooth
        if joint:
            total = total + cfg.dynamics_weight*dynamics + cfg.curvature_weight*curvature + cfg.rate_weight*rate
        return total, dict(reconstruction=reconstruction, overlap=overlap, dynamics=dynamics,
                           curvature=curvature, anchor=anchor, clock_smoothness=smooth, rate=rate)

    def initialize_kinetics(self):
        model, g = self.model, self.model.genes
        grid = self.tensor(np.linspace(0, 1, max(128, self.config.collocation_points)))
        for b in range(model.branches):
            y, dy = model.curve(grid, b)
            y, dy = y.detach().cpu().numpy().astype(float), dy.detach().cpu().numpy().astype(float)
            beta = model.beta[b].detach().cpu().numpy().copy()
            gamma = model.gamma[b].detach().cpu().numpy().copy()
            for gene in range(self.n_selected):
                design = np.column_stack([y[:, gene], -y[:, g+gene]])
                fitted = lsq_linear(design, dy[:, g+gene], bounds=(1e-4, 100))
                if not fitted.success or not np.isfinite(fitted.x).all():
                    raise RuntimeError('Positive-rate initialization failed')
                beta[gene], gamma[gene] = fitted.x
            target = dy[:, :g]+beta*y[:, :g]
            target[:, self.n_selected:] = dy[:, g+self.n_selected:]+gamma[self.n_selected:]*y[:, g+self.n_selected:]
            t = grid.cpu().numpy()
            design = np.column_stack([np.ones(len(t)), t, t*t, t*t*t])
            coefficient = np.linalg.solve(design.T@design + np.diag([1e-8, .01, .01, .01]), design.T@np.maximum(target, 0))
            with torch.no_grad():
                model.beta_raw[b].copy_(inverse_softplus(self.tensor(beta)))
                model.gamma_raw[b].copy_(inverse_softplus(self.tensor(gamma)))
                model.bias[b].copy_(self.tensor(coefficient[0]))
                model.drive[b].copy_(self.tensor(coefficient[1:].T))
            self.kinetic_initialization.append(dict(branch=b, source='warmup-fitted open curve',
                negative_alpha_target_fraction=float((target < 0).mean())))

    def _prediction(self, initial_time, membership, cell_ids, chart_time_override=None):
        t0 = self.tensor(initial_time)
        m = self.tensor(membership)
        sizes = self.tensor(self.charts.membership.sum(0))
        weight = m*sizes[None]
        weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-12)
        n, g = len(t0), self.model.genes
        state = self.y.new_zeros((n, 2*g))
        velocity = torch.zeros_like(state)
        derivative = torch.zeros_like(state)
        with torch.no_grad():
            clocks = (self.clock_values(t0) if chart_time_override is None else self.tensor(chart_time_override)).clamp(0, 1)
            time_value = (clocks*weight).sum(-1)
        for b in range(self.model.branches):
            with torch.no_grad():
                curve = self.model.decode(clocks[:, b], b)
                rhs = self.model.rhs(clocks[:, b], curve, b)
                state += weight[:, b, None]*curve
                velocity += weight[:, b, None]*rhs
            _, tangent = self.model.curve(clocks[:, b], b)
            derivative += weight[:, b, None]*tangent.detach()
        per_chart = clocks.cpu().numpy()
        per_chart[np.asarray(membership) == 0] = np.nan
        return dict(cell_ids=np.asarray(cell_ids, str), genes=self.genes.copy(),
            pseudotime=time_value.cpu().numpy(), initial_graph_time=np.asarray(initial_time),
            chart_times=per_chart, chart_membership=np.asarray(membership),
            S_reconstructed=state[:, g:].cpu().numpy(), U_predicted=state[:, :g].cpu().numpy(),
            velocity_S_kinetic=velocity[:, g:].cpu().numpy(), velocity_U_kinetic=velocity[:, :g].cpu().numpy(),
            velocity_S_curve=derivative[:, g:].cpu().numpy(), velocity_U_curve=derivative[:, :g].cpu().numpy())

    def reference_prediction(self):
        return self._prediction(self.charts.initial_time, self.charts.membership, self.cell_ids)

    def predict(self, S, genes, cell_ids):
        _, positions = self.transform.apply(S, genes)
        ids = names(cell_ids, len(positions), 'query cell IDs')
        if set(ids)&set(self.cell_ids):
            raise ValueError('Query and reference IDs overlap')
        neighbors = min(self.config.neighbors, len(self.cell_ids))
        distances, indices = NearestNeighbors(n_neighbors=neighbors, n_jobs=1).fit(self.positions).kneighbors(positions)
        weights = np.exp(-distances/np.maximum(distances[:, -1:], 1e-8))
        weights /= weights.sum(1, keepdims=True)
        t0 = (self.charts.initial_time[indices]*weights).sum(1)
        membership = (self.charts.membership[indices]*weights[:, :, None]).sum(1)
        return self._prediction(t0, membership, ids)

    def diagnostics(self):
        grid = self.tensor(np.linspace(0, 1, 256))
        roots, endpoints, residuals = [], [], []
        with torch.no_grad():
            clocks = self.clock_values(grid)
        for b in range(self.model.branches):
            y, dy = self.model.curve(grid, b)
            with torch.no_grad():
                residuals.append(float((((dy-self.model.rhs(grid, y, b))/self.model.scale)[:, self.model.active.bool()]).square().mean()))
                roots.append(y[0].detach().cpu().numpy())
                endpoints.append(y[-1].detach().cpu().numpy())
        return dict(topology='open branching atlas', branches=self.model.branches,
            root_state_max_disagreement=float(np.max(np.abs(np.asarray(roots)-roots[0]))),
            minimum_clock_increment=float(torch.diff(clocks, dim=0).min()),
            endpoint_root_state_l2=[float(np.linalg.norm(e-r)) for e,r in zip(endpoints, roots)],
            branch_dynamics_nmse=residuals, training_steps=self.training_steps,
            root_time=0., kinetic_initialization=self.kinetic_initialization,
            inherited_graph_order=True, biological_truth_used_for_training=False)

    def save(self, path, allow_existing=False):
        out = Path(path)
        if out.exists() and not allow_existing:
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
        state = dict(config=asdict(self.config), model=self.model.state_dict(),
            genes=self.genes, cell_ids=self.cell_ids, selected=self.n_selected,
            model_inputs=self.model_inputs, charts=self.charts, transform=self.transform,
            positions=self.positions, S=self.source_S, U=self.source_U, history=self.history,
            training_steps=self.training_steps, kinetic_initialization=self.kinetic_initialization, extra=self.extra_state())
        torch.save(state, out/'checkpoint.pt')
        json_write(out/'config.json', asdict(self.config))

    @classmethod
    def load(cls, path, device='cpu'):
        # Only load locally created experiment checkpoints, never untrusted files.
        state = torch.load(Path(path)/'checkpoint.pt', map_location=device, weights_only=False)
        trainer = cls(cls.config_type(**state['config']), device=device)
        trainer.genes, trainer.cell_ids, trainer.n_selected = state['genes'], state['cell_ids'], state['selected']
        trainer.charts, trainer.transform, trainer.positions = state['charts'], state['transform'], state['positions']
        trainer.model_inputs = inputs = state['model_inputs']
        trainer.model = OpenChartModel(trainer.charts.membership.shape[1], len(trainer.genes), trainer.n_selected,
            trainer.config, *[trainer.tensor(inputs[k]) for k in ['mean', 'scale', 'root_state', 'forbidden']]).to(device)
        trainer.model.load_state_dict(state['model'])
        trainer.model.eval()
        trainer.source_S, trainer.source_U = state['S'], state['U']
        y = np.concatenate([state['U'], state['S']], -1)
        y[:, trainer.n_selected:len(trainer.genes)] = 0
        trainer.y = trainer.tensor(y)
        trainer.t0 = trainer.tensor(trainer.charts.initial_time)
        trainer.membership = trainer.tensor(trainer.charts.membership, torch.bool)
        trainer.initialize_extra()
        trainer.restore_extra(state.get('extra', {}))
        trainer.history, trainer.training_steps = state['history'], state['training_steps']
        trainer.kinetic_initialization = state['kinetic_initialization']
        return trainer
