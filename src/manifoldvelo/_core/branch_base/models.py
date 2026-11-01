"""Open, root-glued lineage curves and positive, order-preserving chart clocks."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..cycle_base.models import inverse_softplus


class OpenClock(nn.Module):
    def __init__(self, branches, knots):
        super().__init__()
        self.log_speed = nn.Parameter(torch.zeros(branches, knots))

    def forward(self, t, branch):
        # Endpoints remain separate. No remainder, sin/cos or periodic roll.
        lengths = torch.softmax(self.log_speed, dim=1)
        lengths = (lengths + 1e-6)/(1 + lengths.shape[1]*1e-6)
        cumulative = torch.cat([lengths.new_zeros(len(lengths), 1), lengths.cumsum(1)], dim=1)
        position = t.clamp(0, 1)*lengths.shape[1]
        index = position.long().clamp(0, lengths.shape[1]-1)
        return cumulative[branch, index] + (position-index)*lengths[branch, index]

    def penalty(self):
        return (self.log_speed[:, 1:] - self.log_speed[:, :-1]).square().mean()


class OpenChartModel(nn.Module):
    def __init__(self, branches, genes, selected, cfg, mean, scale, root_state, forbidden):
        super().__init__()
        self.branches, self.genes, self.selected = branches, genes, selected
        self.register_buffer('scale', scale.clone())
        active = torch.ones(2*genes)
        active[selected:genes] = 0
        self.register_buffer('active', active)
        self.register_buffer('forbidden', forbidden.clone())
        self.clock = OpenClock(branches, cfg.knots)
        self.root_state_raw = nn.Parameter(inverse_softplus((root_state/scale).clamp_min(1e-4)))
        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(1, cfg.hidden), nn.Tanh(),
                            nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, 2*genes))
                            for _ in range(branches)])
        for network in self.decoders:
            with torch.no_grad():
                network[-1].weight.mul_(.05)
                network[-1].bias.zero_()
        self.beta_raw = nn.Parameter(inverse_softplus(torch.full((branches, genes), 2.)))
        ratio = (2*mean[:genes]/mean[genes:].clamp_min(.01)).clamp(.1, 50.)
        ratio[selected:] = 2.
        self.gamma_raw = nn.Parameter(inverse_softplus(ratio[None].repeat(branches, 1)))
        self.W = nn.Parameter(torch.zeros(genes, genes))
        bias = 2*mean[:genes]
        bias[selected:] = 2*mean[genes+selected:]
        self.bias = nn.Parameter(bias[None].repeat(branches, 1))
        # Open polynomial forcing, zero at the root; no periodic phase drive.
        self.drive = nn.Parameter(torch.zeros(branches, genes, 3))
        self.register_buffer('alpha_scale', torch.maximum(2*mean[:genes], ratio*mean[genes:]).clamp_min(.1))

    @property
    def beta(self):
        return F.softplus(self.beta_raw).clamp_min(1e-6)

    @property
    def gamma(self):
        return F.softplus(self.gamma_raw).clamp_min(1e-6)

    def decode(self, t, branch):
        # Each branch has its own open curve; every curve shares Y(0).
        network = self.decoders[branch]
        return F.softplus(self.root_state_raw + t[:, None]*network(t[:, None]))*self.scale*self.active

    def curve(self, t, branch, create_graph=False):
        with torch.enable_grad():
            return torch.autograd.functional.jvp(lambda p: self.decode(p, branch), t,
                       torch.ones_like(t), create_graph=create_graph)

    def rhs(self, t, y, branch):
        u, s = y[:, :self.genes], y[:, self.genes:]
        basis = torch.stack([t, t.square(), t.pow(3)], -1)
        alpha = F.relu(F.linear(s, self.W*(1-self.forbidden), self.bias[branch])
                       + F.linear(basis, self.drive[branch]))
        du = alpha-self.beta[branch]*u
        ds = self.beta[branch]*u-self.gamma[branch]*s
        if self.selected < self.genes:
            k = self.selected
            du = torch.cat([du[:, :k], torch.zeros_like(du[:, k:])], -1)
            ds = torch.cat([ds[:, :k], alpha[:, k:]-self.gamma[branch, k:]*s[:, k:]], -1)
        return torch.cat([du, ds], -1)
