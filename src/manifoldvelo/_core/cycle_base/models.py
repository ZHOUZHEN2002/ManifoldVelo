"""GraphSAGE, a smooth S1 decoder, and the batched TSvelo kinetic equations.

Phase p has period one. State order is [U genes..., S genes...]. Velocities
from ``clock_rhs`` are per *normalized cycle clock*, not per hour.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def circle(p):
    return torch.stack((torch.sin(2 * math.pi * p), torch.cos(2 * math.pi * p)), -1)


def phase(z):
    return torch.remainder(torch.atan2(z[..., 0], z[..., 1]) / (2 * math.pi), 1.)


def circular_delta(p, q):
    """Signed shortest displacement in cycles, including across the 0/1 seam."""
    angle = 2 * math.pi * (p - q)
    return torch.atan2(torch.sin(angle), torch.cos(angle)) / (2 * math.pi)


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class MeanSAGE(nn.Module):
    """Mean GraphSAGE: W_root h_i + W_neighbor mean_{j->i} h_j + b.

    No self edge in the neighbor aggregate. This is the same operation as
    PyG SAGEConv(aggr='mean', normalize=False, project=False).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.neighbor = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, source, target, edge_index):
        src, dst = edge_index
        aggregated = source.new_zeros((len(target), source.shape[-1]))
        aggregated.index_add_(0, dst, source[src])
        counts = source.new_zeros(len(target))
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=source.dtype))
        return self.root(target) + self.neighbor(aggregated / counts.clamp_min(1)[:, None])


class PyGSAGE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError("graph_backend='pyg' requires torch-geometric; "
                              "use graph_backend='native' without that dependency") from exc
        self.conv = SAGEConv((in_dim, in_dim), out_dim, aggr='mean',
                             normalize=False, project=False)

    def forward(self, source, target, edge_index):
        return self.conv((source, target), edge_index, size=(len(source), len(target)))


class CircleEncoder(nn.Module):
    def __init__(self, n_genes, cfg):
        super().__init__()
        self.graph_enabled = cfg.encoder == 'graphsage'
        layer_type = MeanSAGE if cfg.graph_backend == 'native' else PyGSAGE
        dims = [n_genes] + [cfg.hidden_dim] * cfg.encoder_layers
        self.layers = nn.ModuleList([
            layer_type(a, b) if self.graph_enabled else nn.Linear(a, b)
            for a, b in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(dims[-1], 2)

    def _head(self, h):
        raw = self.head(h)
        if not torch.isfinite(raw).all() or torch.any(raw.norm(dim=-1) < 1e-8):
            raise FloatingPointError('Nonfinite or zero-radius circle head; no valid phase can be assigned')
        return F.normalize(raw, dim=-1, eps=1e-8), raw

    def reference(self, x, edges):
        """Keep h^0 ... h^L; reference values never depend on queries."""
        cache = [x]
        for layer in self.layers:
            h = layer(cache[-1], cache[-1], edges) if self.graph_enabled else layer(cache[-1])
            cache.append(F.silu(h))
        z, raw = self._head(cache[-1])
        return z, raw, cache

    def query(self, x, edges, reference_cache):
        h = x
        for depth, layer in enumerate(self.layers):
            h = layer(reference_cache[depth], h, edges) if self.graph_enabled else layer(h)
            h = F.silu(h)
        return self._head(h)


class PeriodicDecoder(nn.Module):
    def __init__(self, mean, scale, n_selected, hidden):
        super().__init__()
        self.register_buffer('scale', scale.clone())
        active = torch.ones_like(mean)
        g = len(mean) // 2
        active[n_selected:g] = 0  # TF-only unspliced channel is not modelled.
        self.register_buffer('active', active)
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, len(mean)))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.copy_(inverse_softplus((mean / scale).clamp_min(1e-4)))

    def forward(self, z):
        return F.softplus(self.net(z)) * self.scale * self.active


class PositiveClock(nn.Module):
    """rho(p)=d tau/dp > 0, with unit mean on a fixed quadrature grid."""
    def __init__(self, cfg):
        super().__init__()
        self.enabled = cfg.learn_clock
        self.bound = cfg.clock_log_bound
        self.register_buffer('grid', torch.arange(cfg.clock_grid, dtype=torch.float32) / cfg.clock_grid)
        self.net = None
        if self.enabled:
            self.net = nn.Sequential(nn.Linear(2, cfg.clock_hidden), nn.Tanh(), nn.Linear(cfg.clock_hidden, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _positive(self, p):
        return torch.exp(self.bound * torch.tanh(self.net(circle(p)).squeeze(-1)))

    def forward(self, p):
        if not self.enabled:
            return torch.ones_like(p)
        return self._positive(p) / self._positive(self.grid).mean()


class TSveloDynamics(nn.Module):
    """Original TSvelo equations plus explicit phase drive and optional residual.

    Mask convention: W_0_mask[g,tf] == 1 FORBIDS that edge. Effective rates
    are softplus-parametrized (strictly positive), unlike the legacy ReLU
    parametrization. At equal positive rates the RHS is identical.
    """
    def __init__(self, n_genes, n_selected, cfg, W_0_mask, initial, alpha_scale):
        super().__init__()
        self.n_genes, self.n_selected = n_genes, n_selected
        self.residual_bound = cfg.residual_bound
        self.register_buffer('W_0_mask', W_0_mask.clone())
        self.register_buffer('alpha_scale', alpha_scale.clone())
        for key in ('beta', 'gamma'):
            value = initial[key].clone()
            self.register_parameter(key + '_raw', nn.Parameter(inverse_softplus(value)))
            self.register_buffer(key + '_initial', value)
        self.W = nn.Parameter(initial['W'].clone())
        self.W_bias = nn.Parameter(initial['W_bias'].clone())
        self.phase_W = nn.Parameter(initial['phase_W'].clone())
        self.residual_net = None
        if cfg.residual:
            self.residual_net = nn.Sequential(
                nn.Linear(2, cfg.residual_hidden), nn.Tanh(),
                nn.Linear(cfg.residual_hidden, cfg.residual_rank), nn.Tanh(),
                nn.Linear(cfg.residual_rank, n_genes, bias=False))
            nn.init.zeros_(self.residual_net[-1].weight)

    @property
    def beta(self):
        return F.softplus(self.beta_raw).clamp_min(1e-8)

    @property
    def gamma(self):
        return F.softplus(self.gamma_raw).clamp_min(1e-8)

    def residual(self, p):
        if self.residual_net is None:
            return p.new_zeros(p.shape + (self.n_genes,))
        return self.residual_bound * self.alpha_scale * torch.tanh(self.residual_net(circle(p)))

    def clock_rhs(self, p, y):
        u, s = y[..., :self.n_genes], y[..., self.n_genes:]
        drive = F.linear(s, self.W * (1 - self.W_0_mask), self.W_bias)
        alpha = F.relu(drive + F.linear(circle(p), self.phase_W) + self.residual(p))
        du = alpha - self.beta * u
        ds = self.beta * u - self.gamma * s
        if self.n_selected < self.n_genes:
            cut = self.n_selected
            du = torch.cat((du[..., :cut], torch.zeros_like(du[..., cut:])), -1)
            ds = torch.cat((ds[..., :cut], alpha[..., cut:] - self.gamma[cut:] * s[..., cut:]), -1)
        return torch.cat((du, ds), -1)


class ManifoldModel(nn.Module):
    def __init__(self, n_genes, n_selected, cfg, mean, scale, W_0_mask, initial):
        super().__init__()
        self.n_genes = n_genes
        self.encoder = CircleEncoder(n_genes, cfg)
        self.decoder = PeriodicDecoder(mean, scale, n_selected, cfg.decoder_hidden)
        self.clock = PositiveClock(cfg)
        alpha_scale = torch.maximum(initial['beta'] * mean[:n_genes],
                                    initial['gamma'] * mean[n_genes:]).clamp_min(0.1)
        self.dynamics = TSveloDynamics(n_genes, n_selected, cfg, W_0_mask, initial, alpha_scale)
        self.register_buffer('state_scale', scale.clone())

    def decode_phase(self, p):
        return self.decoder(circle(p))

    def curve(self, p, create_graph=False):
        # The decoder is row-independent; this JVP gives each row's dY/dp
        # without looping over genes or constructing a full Nx2GxN Jacobian.
        with torch.enable_grad():
            y, dy_dp = torch.autograd.functional.jvp(self.decode_phase, p, torch.ones_like(p),
                                                    create_graph=create_graph)
        return y, dy_dp

    def phase_rhs(self, p, y):
        return self.clock(p)[..., None] * self.dynamics.clock_rhs(p, y)
