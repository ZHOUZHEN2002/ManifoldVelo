import numpy as np
import torch
from torch import nn

class MonotoneCircle(nn.Module):
    def __init__(self, phases, stride=4):
        super().__init__()
        points = np.sort(np.asarray(phases) % 1)
        knots = np.unique(np.r_[0., points[::stride], 1.].astype(np.float32))
        width = np.diff(knots)
        # Retain original spectral timing at initialization, not uniform rank.
        self.register_buffer('knots', torch.tensor(knots))
        self.register_buffer('width', torch.tensor(width))
        self.log_multiplier = nn.Parameter(torch.zeros(len(width)))
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, p):
        p = p.remainder(1)
        delta = self.width * torch.exp(self.log_multiplier.clamp(-12,12))
        delta = delta / delta.sum()
        delta = (delta + 1e-7) / (1 + len(delta)*1e-7)
        cumulative = torch.cat((delta.new_zeros(1), delta.cumsum(0)))
        index = torch.searchsorted(self.knots, p.contiguous(), right=True).sub(1).clamp(0, len(delta)-1)
        fraction = (p-self.knots[index])/self.width[index]
        return (cumulative[index]+fraction*delta[index]+self.offset).remainder(1)

    def smoothness(self):
        # Weak smoothness of log density on adjacent adaptive chart intervals.
        return (self.log_multiplier-self.log_multiplier.roll(1)).square().mean()

def wrap(x):
    return (np.asarray(x) + .5) % 1 - .5

class Stability:
    """Operational local convergence, NOT proof of a global optimum.

    Compares fixed-grid training diagnostics every check, requiring three
    consecutive stable windows, as well as a minimum training duration.
    No small learning rate, elapsed time or iteration cap counts as convergence.
    """
    def __init__(self, loss_tol=.005, phase_tol=.002, curve_tol=.01,
                 param_tol=.02, checks=3, minimum=1500):
        self.settings = dict(loss_tol=loss_tol, phase_tol=phase_tol,
            curve_tol=curve_tol, param_tol=param_tol, checks=checks, minimum=minimum)
        self.previous = None
        self.streak = 0
        self.records = []

    def update(self, step, loss, phase, curve, params):
        current = dict(loss=float(loss), phase=np.asarray(phase).copy(),
                       curve=np.asarray(curve).copy(), params=np.asarray(params).copy())
        row = dict(step=int(step), loss=float(loss), stable=False, streak=0, converged=False)
        if self.previous is not None:
            prev = self.previous
            row.update(loss_relative_change=abs(loss - prev['loss']) / max(abs(loss), abs(prev['loss']), 1.),
                phase_mean_change=float(np.mean(np.abs(wrap(current['phase'] - prev['phase'])))),
                curve_relative_change=float(np.linalg.norm(current['curve'] - prev['curve']) /
                                            max(np.linalg.norm(prev['curve']), 1e-8)),
                parameter_relative_change=float(np.linalg.norm(current['params'] - prev['params']) /
                                                max(np.linalg.norm(prev['params']), 1e-8)))
            s = self.settings
            row['stable'] = bool(row['loss_relative_change'] <= s['loss_tol'] and
                row['phase_mean_change'] <= s['phase_tol'] and
                row['curve_relative_change'] <= s['curve_tol'] and
                row['parameter_relative_change'] <= s['param_tol'])
            self.streak = self.streak + 1 if row['stable'] else 0
            row['streak'] = self.streak
            row['converged'] = bool(self.streak >= s['checks'] and step >= s['minimum'])
        self.previous = current
        self.records.append(row)
        return row

    def state(self):
        return dict(settings=self.settings, previous=self.previous,
                    streak=self.streak, records=self.records)

    @classmethod
    def restore(cls, state):
        obj = cls(**state['settings'])
        obj.previous, obj.streak, obj.records = state['previous'], state['streak'], state['records']
        return obj

