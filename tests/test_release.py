import copy
import numpy as np
import pytest
import torch
import manifoldvelo as mv
from manifoldvelo._core.cycle_base.synthetic import generate
from manifoldvelo._core.temporal import MonotoneCircle


@pytest.fixture(scope='module')
def rna():
    torch.set_num_threads(1)
    data = generate(cells=160, genes=6, seed=7)
    take = data['train_mask']
    train = dict(S=data['S'][take], U=data['U'][take], genes=data['genes'],
                 cell_ids=data['cell_ids'][take], W_0_mask=np.eye(6, dtype=np.float32))
    query = dict(S=data['S'][~take][:8], genes=data['genes'], cell_ids=data['cell_ids'][~take][:8])
    return train, query


def finite_global(pred):
    time = pred.get('phase', pred.get('pseudotime'))
    assert np.isfinite(time).all()
    for key in ['velocity_S_kinetic', 'velocity_U_kinetic', 'S_reconstructed', 'U_predicted']:
        assert np.isfinite(pred[key]).all()


def test_cycle_fit_reload_and_query_isolation(tmp_path, rna):
    train, query = rna
    cfg = dict(warmup_steps=4, joint_steps=5, hidden_dim=16, decoder_hidden=16,
               neighbors=8, graph_pcs=6, collocation_points=8, batch_size=64)
    model = mv.fit(**train, family='cycle', root_index=0, config=cfg, output=tmp_path/'cycle')
    before = {key: value.detach().clone() for key, value in model.model.state_dict().items()}
    p = mv.predict(model, **query)
    finite_global(p)
    single = mv.predict(model, query['S'][:1], query['genes'], query['cell_ids'][:1])
    reordered = mv.predict(model, query['S'][:, ::-1], query['genes'][::-1], query['cell_ids'])
    np.testing.assert_allclose(single['phase'], p['phase'][:1], atol=2e-6)
    np.testing.assert_allclose(reordered['velocity_S_kinetic'], p['velocity_S_kinetic'], atol=2e-5)
    for key, value in before.items():
        torch.testing.assert_close(value, model.model.state_dict()[key], rtol=0, atol=0)
    restored = mv.load_reference(tmp_path/'cycle')
    p2 = mv.predict(restored, **query)
    np.testing.assert_allclose(p2['phase'], p['phase'], atol=2e-6)
    np.testing.assert_allclose(p2['velocity_S_kinetic'], p['velocity_S_kinetic'], atol=2e-5)
    roots = model.model.decode_phase(torch.tensor([0., 1.]))
    torch.testing.assert_close(roots[0], roots[1], atol=2e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='overlap'):
        mv.predict(model, train['S'][:1], train['genes'], train['cell_ids'][:1])


def test_open_fit_shared_origin_and_reload(tmp_path, rna):
    train, query = rna
    n = len(train['S'])
    t = np.linspace(0, 1, n, dtype=np.float32)
    membership = np.zeros((n, 2), bool)
    membership[:20] = True
    membership[20:n//2, 0] = True
    membership[n//2:, 1] = True
    charts = mv.BranchCharts(t, membership, [['root', 'a'], ['root', 'b']], 0,
                             {'source': 'controlled software-test chart fixture'})
    config = dict(warmup_steps=4, joint_steps=5, hidden=16, knots=8,
                  neighbors=8, graph_pcs=6, collocation_points=8,
                  projection_every=2, projection_points=32)
    model = mv.fit(**train, family='open', root_index=0, charts=charts,
                   config=config, output=tmp_path/'open')
    p = mv.predict(model, **query)
    finite_global(p)
    z = torch.zeros(1)
    torch.testing.assert_close(model.model.decode(z, 0), model.model.decode(z, 1), atol=0, rtol=0)
    assert model.reference_prediction()['pseudotime'][0] == 0
    p2 = mv.predict(mv.load_reference(tmp_path/'open'), **query)
    np.testing.assert_allclose(p2['velocity_S_kinetic'], p['velocity_S_kinetic'], atol=2e-5)
    single = mv.predict(model, query['S'][:1], query['genes'], query['cell_ids'][:1])
    np.testing.assert_allclose(single['velocity_S_kinetic'], p['velocity_S_kinetic'][:1], atol=2e-5)


def test_monotone_map_and_multipeak_reload(tmp_path, rna):
    train, query = rna
    mapping = MonotoneCircle(np.linspace(0, 1, 40, endpoint=False))
    p = mapping(torch.linspace(0, .999, 50)).detach().numpy()
    assert (np.diff(p) > 0).all()
    model = mv.fit(**train, family='multipeak', root_index=0,
                   output=tmp_path/'multipeak', max_steps=2)
    predicted = mv.predict(model, **query)
    finite_global(predicted)
    restored = mv.predict(mv.load_reference(tmp_path/'multipeak'), **query)
    np.testing.assert_allclose(restored['phase'], predicted['phase'], atol=2e-6)
    np.testing.assert_allclose(restored['velocity_S_kinetic'], predicted['velocity_S_kinetic'], atol=2e-5)


def test_root_selection_is_reproducible(rna):
    train, _ = rna
    args = {key: train[key] for key in ['S', 'U', 'genes', 'cell_ids']}
    a = mv.infer_root(**args, bootstraps=3)
    b = mv.infer_root(**args, bootstraps=3)
    assert a == b
    assert args['cell_ids'][a['root_index']] == a['root_id']
    assert 0 <= a['bootstrap_support'] <= 1


def test_invalid_inputs_fail_before_output_creation(tmp_path, rna):
    train, _ = rna
    broken = copy.deepcopy(train)
    broken['U'][0, 0] = np.nan
    with pytest.raises(ValueError):
        mv.fit(**broken, family='cycle', root_index=0, output=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
