"""Experimental GraphSAGE / S1 / TSvelo dynamical-manifold framework."""
from .config import ManifoldConfig
from .trainer import ManifoldTrainer

__all__ = ['ManifoldConfig', 'ManifoldTrainer']
__version__ = '0.1.0'
