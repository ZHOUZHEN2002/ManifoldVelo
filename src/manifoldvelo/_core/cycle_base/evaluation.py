"""Phase truth is used ONLY after fitting, including train-only gauge fixing."""
import numpy as np


def wrap(p):
    return (np.asarray(p) + .5) % 1 - .5


def fit_phase_gauge(train_predicted, train_truth):
    """Fit origin/reflection on training cells only; freeze for test scoring.

    A reflection is reported, not hidden: phase-gauge invariance is not evidence
    that the model identified biological direction. No nonlinear time warping.
    """
    candidates = []
    for sign in (1, -1):
        rotation = np.mean(np.exp(2j * np.pi * (train_truth - sign * train_predicted)))
        offset = float(np.angle(rotation) / (2 * np.pi))
        candidates.append((float(np.mean(wrap(sign * train_predicted + offset - train_truth)**2)), sign, offset))
    _, sign, offset = min(candidates)
    return dict(sign=sign, offset_cycles=offset, fitted_on='training truth after model training',
                nonlinear_time_warp=False)


def apply_phase_gauge(predicted, gauge):
    return (gauge['sign'] * np.asarray(predicted) + gauge['offset_cycles']) % 1


def phase_scores(predicted, truth, gauge):
    error = wrap(apply_phase_gauge(predicted, gauge) - truth)
    return dict(circular_mae_cycles=float(np.abs(error).mean()),
                circular_mae_degrees=float(np.abs(error).mean() * 360),
                circular_rmse_cycles=float(np.sqrt(np.mean(error**2))),
                circular_cosine_agreement=float(np.cos(2 * np.pi * error).mean()))


def prediction_nmse(predicted, observed, training_observed, scale_floor=.1):
    scale = np.maximum(np.asarray(training_observed).std(0), scale_floor)
    return float(np.mean(((predicted - observed) / scale)**2))
