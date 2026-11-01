"""Open neural-curve M steps and root-constrained nearest-curve time E steps.

This reuses the key *idea* of the original TSvelo time update, but fits neural
open curves with a kinetic compatibility loss rather than the original ODE
solver. No native prediction or biological time label enters these updates.
"""
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from ..cycle_base.graph import names
from .config import BranchConfig
from .trainer import BranchManifoldTrainer


@dataclass
class EMBranchConfig(BranchConfig):
    projection_every: int = 200
    projection_points: int = 1000

    def validate(self):
        super().validate()
        if self.warmup_steps < 1 or self.projection_every < 1 or self.projection_points < 16:
            raise ValueError('EM needs warmup, a positive projection interval and >=16 grid points')
        return self


class EMBranchManifoldTrainer(BranchManifoldTrainer):
    config_type = EMBranchConfig

    def protocol_details(self):
        return dict(clock='warmup monotone open clocks; joint training alternates neural curve fitting and nearest-curve time projection',
            output_time='branch-size weighted inferred cell times, with all chart times also exported',
            fit_scale='per-chart U/S standard deviations, matching the native fitting metric',
            limitation='PAGA defines lineage support; internal junctions use soft overlap consistency; no automatic mixed-topology discovery',
            lineage_order_can_change=True, root_time_fixed_to_zero=True,
            transcription_prior_scale='updated from train-curve initialized RNA rates',
            native_ode_solver_used=False, native_fitted_time_used=False)

    def initialize_extra(self):
        self.assignments = None
        self.projection_history = []
        # Match native per-lineage U/S standardization in the fitting metric,
        # while decoding and evaluating the RNA equations in original units.
        self.branch_scales = torch.stack([self.y[self.membership[:, b]].std(0, unbiased=False).clamp_min(self.config.state_scale_floor)
                                         for b in range(self.model.branches)])

    def state_scale(self, branch):
        return self.branch_scales[branch]

    def training_clock_values(self, batch):
        return super().training_clock_values(batch) if self.assignments is None else self.assignments[batch]

    def extra_state(self):
        return dict(assignments=self.assignments, projection_history=self.projection_history,
                    branch_scales=self.branch_scales)

    def restore_extra(self, state):
        self.assignments = state.get('assignments')
        self.projection_history = state.get('projection_history', [])
        if 'branch_scales' in state:
            self.branch_scales = state['branch_scales'].to(self.device)
        if self.assignments is not None:
            self.assignments = self.assignments.to(self.device)
            for p in self.model.clock.parameters():
                p.requires_grad_(False)

    def initialize_kinetics(self):
        super().initialize_kinetics()
        # Normalize transcription regularization in the units of the newly
        # initialized rates, instead of the arbitrary pre-warmup beta=2.
        with torch.no_grad():
            g = self.model.genes
            mean = self.y.mean(0)
            scale = torch.maximum(self.model.beta*mean[:g], self.model.gamma*mean[g:]).mean(0).clamp_min(.1)
            self.model.alpha_scale.copy_(scale)

    def after_step(self, step):
        if step == self.config.warmup_steps:
            with torch.no_grad():
                self.assignments = self.clock_values(self.t0).detach().clone()
            for parameter in self.model.clock.parameters():
                parameter.requires_grad_(False)
        if step > self.config.warmup_steps and ((step-self.config.warmup_steps)%self.config.projection_every == 0):
            self.project_training_times(step)

    def project_training_times(self, step):
        model, cfg = self.model, self.config
        grid = self.tensor(np.linspace(0, 1, cfg.projection_points))
        active = model.active.bool()
        before, after, moved = [], [], []
        with torch.no_grad():
            for b in range(model.branches):
                ids = torch.where(self.membership[:, b])[0]
                scale = self.state_scale(b)[active]
                curve = model.decode(grid, b)[:, active]/scale
                sq_curve = curve.square().mean(-1)
                for block in ids.split(1024):
                    observed = self.y[block][:, active]/scale
                    costs = observed.square().mean(-1, keepdim=True)+sq_curve[None]-2*(observed@curve.T)/curve.shape[1]
                    costs = costs.clamp_min(0)+cfg.anchor_weight*(grid[None]-self.t0[block, None]).square()
                    best_cost, index = costs.min(-1)
                    old_time = self.assignments[block, b].clone()
                    old_curve = model.decode(old_time, b)[:, active]/scale
                    old_cost = (observed-old_curve).square().mean(-1)+cfg.anchor_weight*(old_time-self.t0[block]).square()
                    improve = best_cost < old_cost
                    improve &= block != self.charts.root_index
                    new_time = torch.where(improve, grid[index], old_time)
                    self.assignments[block, b] = new_time
                    before.append(old_cost.mean())
                    after.append(torch.where(improve, best_cost, old_cost).mean())
                    moved.append((new_time-old_time).abs().mean())
            self.assignments[self.charts.root_index] = 0
        row = dict(step=step, before=float(torch.stack(before).mean()),
            after=float(torch.stack(after).mean()), mean_time_update=float(torch.stack(moved).mean()))
        if row['after'] > row['before']+1e-6:
            raise AssertionError('Time projection increased its conditional objective')
        self.projection_history.append(row)

    def reference_prediction(self):
        times = None if self.assignments is None else self.assignments.cpu().numpy()
        return self._prediction(self.charts.initial_time, self.charts.membership, self.cell_ids,
                                chart_time_override=times)

    def predict(self, S, genes, cell_ids):
        _, positions = self.transform.apply(S, genes)
        ids = names(cell_ids, len(positions), 'query cell IDs')
        if set(ids)&set(self.cell_ids):
            raise ValueError('Query and reference IDs overlap')
        count = min(self.config.neighbors, len(self.cell_ids))
        distances, index = NearestNeighbors(n_neighbors=count, n_jobs=1).fit(self.positions).kneighbors(positions)
        weights = np.exp(-distances/np.maximum(distances[:, -1:], 1e-8))
        weights /= weights.sum(1, keepdims=True)
        t0 = (self.charts.initial_time[index]*weights).sum(1)
        membership = (self.charts.membership[index]*weights[:, :, None]).sum(1)
        # Condition on S only. U_query and true query times are never accepted.
        order = {str(g):j for j,g in enumerate(genes)}
        observed = self.tensor(np.asarray(S)[:, [order[str(g)] for g in self.genes]])
        grid = self.tensor(np.linspace(0, 1, self.config.projection_points))
        time_by_chart = np.zeros((len(S), self.model.branches), np.float32)
        with torch.no_grad():
            for b in range(self.model.branches):
                scale = self.state_scale(b)[self.model.genes:]
                curve = self.model.decode(grid, b)[:, self.model.genes:]/scale
                for start in range(0, len(S), 1024):
                    block = slice(start, start+1024)
                    y = observed[block]/scale
                    cost = (y.square().mean(-1, keepdim=True)+curve.square().mean(-1)[None]-2*(y@curve.T)/curve.shape[1]).clamp_min(0)
                    cost += self.config.anchor_weight*(grid[None]-self.tensor(t0[block])[:, None]).square()
                    time_by_chart[block, b] = grid[cost.argmin(-1)].cpu().numpy()
        return self._prediction(t0, membership, ids, chart_time_override=time_by_chart)

    def diagnostics(self):
        result = super().diagnostics()
        result.update(time_inference='alternating nearest open-curve projections using observed training S/U',
            inherited_graph_order=False, projection_history=self.projection_history,
            graph_used_for='rooted lineage support, initialization, soft time prior, query branch membership',
            query='S-only nearest-curve projection with training-reference graph prior')
        return result
