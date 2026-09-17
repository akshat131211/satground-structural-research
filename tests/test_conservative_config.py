from pathlib import Path

from satground.runs import read_config


def test_revision_changes_only_common_learning_rate_and_named_losses():
    root = Path(__file__).parents[1]
    configs = {n: read_config(root / f'configs/conservative500/{n}.yaml') for n in ('A1', 'A2', 'A3')}
    allowed = {'experiment', 'region_weight', 'boundary_weight'}
    shared = [{k: v for k, v in c.items() if k not in allowed} for c in configs.values()]
    assert shared[0] == shared[1] == shared[2]
    assert configs['A1']['learning_rate'] == .00003
    assert configs['A1']['steps'] == 500
    assert configs['A1']['seed'] == 17
    assert configs['A1']['data_review_mode'] == 'exploratory'
    assert [(c['region_weight'], c['boundary_weight']) for c in configs.values()] == [(0, 0), (.5, 0), (.5, .5)]
    previous = read_config(root / 'configs/exploratory500/A1.yaml')
    differences = {k for k in previous if previous[k] != configs['A1'][k]}
    assert differences == {'learning_rate', 'exploratory_reason'}
