from pathlib import Path

import pytest
import yaml

from satground.common import ROOT, ResearchError
from satground.runs import read_config


def test_field_experiments_preserve_original_protocol(tmp_path):
    original = read_config(ROOT / 'configs/learning100/A3.yaml')
    for experiment, mode in [('GJ', 'joint'), ('GD', 'density')]:
        cfg = dict(original, experiment=experiment, field_mode=mode)
        path = tmp_path / f'{experiment}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        assert read_config(path) == cfg
        cfg['field_mode'] = 'appearance' if mode == 'joint' else 'joint'
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ResearchError, match='explicit shared-sampling'):
            read_config(path)
    path = tmp_path / 'original.yaml'
    path.write_text(yaml.safe_dump(dict(original, field_mode='joint')))
    with pytest.raises(ResearchError, match='original A experiments'):
        read_config(path)


def test_geometry_controls_have_matching_budgets_and_supervision():
    a = read_config(ROOT / 'configs/geometry100/GJ.yaml')
    b = read_config(ROOT / 'configs/geometry100/GD.yaml')
    assert {k: v for k, v in a.items() if k not in ('experiment', 'field_mode')} == {
        k: v for k, v in b.items() if k not in ('experiment', 'field_mode')}
    assert a['steps'] == 100
