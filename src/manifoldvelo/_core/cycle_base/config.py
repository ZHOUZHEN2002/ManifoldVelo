"""Small, serializable configuration; no dataset-specific choices or labels."""
from dataclasses import dataclass, asdict
import math


@dataclass
class ManifoldConfig:
    encoder: str = 'graphsage'
    graph_backend: str = 'native'
    hidden_dim: int = 128
    encoder_layers: int = 2
    decoder_hidden: int = 64
    neighbors: int = 15
    graph_pcs: int = 20
    feature_scale_floor: float = 0.1
    feature_clip: float = 8.0
    state_scale_floor: float = 0.1
    residual: bool = False
    residual_hidden: int = 16
    residual_rank: int = 8
    residual_bound: float = 0.2
    learn_clock: bool = False
    clock_hidden: int = 16
    clock_log_bound: float = 0.5
    clock_grid: int = 128
    collocation_points: int = 64
    warmup_steps: int = 100
    warmup_anchor_weight: float = 1.0
    initialize_kinetics: bool = True
    kinetic_ridge: float = 0.01
    joint_steps: int = 200
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    gradient_clip: float = 5.0
    dynamics_weight: float = 0.1
    geometry_weight: float = 0.05
    anchor_weight: float = 0.02
    radius_weight: float = 0.001
    residual_weight: float = 0.01
    curvature_weight: float = 0.0001
    rate_weight: float = 0.001
    clock_weight: float = 0.01
    negative_margin: float = 1.0
    seed: int = 0
    root_weight: float = 0.0
    root_phase: float = 0.05

    def validate(self):
        if self.encoder not in ('graphsage', 'mlp'):
            raise ValueError('encoder must be graphsage or mlp')
        if self.graph_backend not in ('native', 'pyg'):
            raise ValueError('graph_backend must be native or pyg')
        if not 0 <= self.root_phase < 1:
            raise ValueError('root_phase must be in [0,1)')
        for key in ('hidden_dim','encoder_layers','decoder_hidden','neighbors','graph_pcs',
                    'residual_hidden','residual_rank','clock_hidden','clock_grid',
                    'collocation_points','batch_size'):
            value=getattr(self,key)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(key+' must be a positive integer')
        if self.clock_grid<16 or self.collocation_points<8:
            raise ValueError('clock_grid >=16 and collocation_points >=8 required')
        for key in ('warmup_steps','joint_steps','seed'):
            value=getattr(self,key)
            if not isinstance(value,int) or isinstance(value,bool) or value<0:
                raise ValueError(key+' must be a nonnegative integer')
        if self.warmup_steps+self.joint_steps==0:raise ValueError('At least one training step required')
        for key,value in asdict(self).items():
            if isinstance(value,float) and (not math.isfinite(value) or value<0):
                raise ValueError(key+' must be finite and nonnegative')
        for key in ('feature_scale_floor','feature_clip','state_scale_floor','learning_rate','gradient_clip'):
            if getattr(self,key)<=0:raise ValueError(key+' must be positive')
        return self

    def to_dict(self):return asdict(self)
