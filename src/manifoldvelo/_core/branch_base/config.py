from dataclasses import asdict, dataclass
import math


@dataclass
class BranchConfig:
    hidden: int = 96
    knots: int = 48
    neighbors: int = 30
    graph_pcs: int = 20
    feature_scale_floor: float = .1
    feature_clip: float = 8.
    state_scale_floor: float = .1
    warmup_steps: int = 800
    joint_steps: int = 3200
    batch_size: int = 2048
    collocation_points: int = 96
    learning_rate: float = .002
    final_learning_rate: float = .0001
    dynamics_weight: float = .01
    anchor_weight: float = .05
    overlap_weight: float = .1
    smoothness_weight: float = .001
    curvature_weight: float = .00001
    rate_weight: float = .0001
    gradient_clip: float = 5.
    seed: int = 0

    def validate(self):
        for key in ['hidden', 'knots', 'neighbors', 'graph_pcs', 'batch_size', 'collocation_points']:
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(key + ' must be a positive integer')
        for key in ['seed', 'warmup_steps', 'joint_steps']:
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 0:
                raise ValueError(key + ' must be a nonnegative integer')
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError(key + ' must be finite and nonnegative')
        if self.knots < 4 or self.collocation_points < 8 or self.warmup_steps + self.joint_steps < 1:
            raise ValueError('Need >=4 knots, >=8 collocation points and positive training budget')
        if min(self.learning_rate, self.final_learning_rate, self.state_scale_floor,
               self.feature_scale_floor, self.feature_clip, self.gradient_clip) <= 0:
            raise ValueError('Learning rates, scales, clip bounds must be positive')
        return self

    def to_dict(self):
        return asdict(self)
