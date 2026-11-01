"""Shared circular root supervision; no cell-type, age or endpoint labels."""
import math
import numpy as np
import torch
import pyro
from pyro import poutine

ROOT_WEIGHT = .1
ROOT_PHASE = .05


def root_penalty(p, mask, target=ROOT_PHASE):
    if mask is None or mask.dtype != torch.bool or mask.shape != p.shape or not bool(mask.any()):
        raise ValueError('Root supervision requires a nonempty boolean mask matching cell phases')
    return (1 - torch.cos(2 * math.pi * (p[mask] - target))).mean()


def align_initial_phase(p, roots, target=ROOT_PHASE):
    p, roots = np.asarray(p), np.asarray(roots)
    if p.ndim != 1 or roots.dtype != bool or roots.shape != p.shape or not roots.any():
        raise ValueError('Invalid root mask for circular initialization')
    center = np.mean(np.exp(2j * np.pi * p[roots]))
    if abs(center) < 1e-8: raise ValueError('Root phases have no identifiable circular mean')
    shift = target - np.angle(center) / (2*np.pi)
    return ((p + shift) % 1).astype(np.float32), float(shift), float(abs(center))


def normalized_model(native, normalizer, roots=None, weight=0., target=ROOT_PHASE):
    if normalizer <= 0 or weight < 0: raise ValueError('Invalid objective weights')
    def model(mp):
        with poutine.scale(scale=1./normalizer):
            trace = poutine.trace(native).get_trace(mp)
        if weight:
            xy = trace.nodes['ϕxy']['value']
            p = torch.atan2(xy[..., 1], xy[..., 0]) / (2 * math.pi)
            pyro.factor('biological_root_supervision', -weight * root_penalty(p, roots, target))
    return model


def normalized_guide(native, normalizer):
    def guide(mp):
        with poutine.scale(scale=1./normalizer):
            return native(mp)
    return guide
