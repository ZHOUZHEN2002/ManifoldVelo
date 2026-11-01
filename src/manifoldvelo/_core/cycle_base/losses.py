"""Keep fit quality, geometry, and dynamical agreement separate."""
import math
import torch
from .models import circular_delta, phase
from ..rooting import root_penalty


def masked_mse(predicted, observed, mask, scale):
    # Equal weight for each active gene/channel, independent of missingness.
    errors = ((predicted - observed) / scale).square() * mask
    count = mask.sum(0)
    active = count > 0
    return (errors.sum(0)[active] / count[active]).mean()


def graph_loss(z, positive, negative, margin):
    p = phase(z)
    pos = circular_delta(p[positive[0]], p[positive[1]]) * (2 * math.pi)
    value = pos.square().mean()
    if negative.shape[1]:
        neg = (circular_delta(p[negative[0]], p[negative[1]]) * (2 * math.pi)).abs()
        value = value + torch.relu(margin - neg).square().mean()
    return value


def objective(model, x, y, mask, edges, negative, initial_z, batch, p_grid, cfg, joint, root_mask=None):
    z, raw, _ = model.encoder.reference(x, edges)
    predicted = model.decoder(z[batch])
    zero = predicted.sum() * 0
    terms = {
        'reconstruction': masked_mse(predicted, y[batch], mask[batch], model.state_scale),
        'geometry': graph_loss(z, edges, negative, cfg.negative_margin),
        'anchor': (2 * math.pi * circular_delta(phase(z), phase(initial_z))).square().mean(),
        'radius': (raw.square().sum(-1) - 1).square().mean(),
        'dynamics': zero, 'residual': zero, 'curvature': zero, 'rate_prior': zero, 'clock': zero,
        'root': root_penalty(phase(z), root_mask, cfg.root_phase) if cfg.root_weight else zero,
    }
    if joint:
        curve, dy_dp = model.curve(p_grid, create_graph=True)
        rho = model.clock(p_grid)
        active = model.decoder.active.bool()
        difference = (dy_dp / rho[:, None] - model.dynamics.clock_rhs(p_grid, curve)) / model.state_scale
        terms['dynamics'] = difference[:, active].square().mean()
        # Uniform grid differences include the last-to-first periodic seam.
        curvature = (dy_dp.roll(-1, 0) - dy_dp) * len(p_grid) / model.state_scale
        terms['curvature'] = curvature[:, active].square().mean()
        dyn = model.dynamics
        terms['residual'] = (dyn.residual(p_grid) / dyn.alpha_scale).square().mean()
        terms['rate_prior'] = (torch.log(dyn.beta / dyn.beta_initial).square().mean()
                               + torch.log(dyn.gamma / dyn.gamma_initial).square().mean()
                               + (dyn.W * (1 - dyn.W_0_mask)).square().mean()
                               + (dyn.phase_W / dyn.alpha_scale[:, None]).square().mean())
        terms['clock'] = torch.log(rho).square().mean()
    anchor_weight = cfg.anchor_weight if joint else cfg.warmup_anchor_weight
    total = (terms['reconstruction'] + cfg.geometry_weight * terms['geometry']
             + anchor_weight * terms['anchor'] + cfg.radius_weight * terms['radius']
             + cfg.dynamics_weight * terms['dynamics'] + cfg.residual_weight * terms['residual']
             + cfg.curvature_weight * terms['curvature'] + cfg.rate_weight * terms['rate_prior']
             + cfg.clock_weight * terms['clock'])
    total = total + cfg.root_weight * terms['root']
    return total, terms
