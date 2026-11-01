from dataclasses import asdict
import numpy as np
import torch
from .common import write

def save_model(m, path):
    """Native checkpoint format; omit the optional, unrelated ODE rollout audit."""
    path.mkdir(exist_ok=False)
    def cpu(v):
        return v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v)
    transform = {k: cpu(v) if isinstance(v, np.ndarray) and k != 'genes' else v
                 for k, v in asdict(m.transform).items()}
    transform['genes'] = m.genes.tolist()
    graph = asdict(m.graph)
    for key in ['positions', 'edge_index', 'initial_z']:
        graph[key] = cpu(graph[key])
    bundle = dict(format_version=1, config=m.config.to_dict(), genes=m.genes.tolist(),
        cell_ids=m.cell_ids.tolist(), n_selected=m.n_selected, kinetic_initialization=m.kinetic_initialization,
        transform=transform, graph=graph, state_mean=cpu(m.state_mean), state_scale=cpu(m.model.state_scale),
        W_0_mask=cpu(m.W_0_mask), initial={k: cpu(v) for k, v in m.initial.items()},
        x=cpu(m.x), y=cpu(m.y), mask=cpu(m.mask), root_mask=cpu(m.root_mask),
        model_state={k: cpu(v) for k, v in m.model.state_dict().items()})
    torch.save(bundle, path / 'checkpoint.pt')
    write(path / 'config.json', m.config.to_dict())
    write(path / 'history.json', m.history)

