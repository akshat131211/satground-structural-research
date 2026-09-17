import copy
import importlib
from pathlib import Path

import pytest

from satground.common import ResearchError


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    return importlib.import_module('report_conservative_trajectory')


def row(sid, boundary):
    return dict(sample_id=sid, geo_group=sid, boundary_error=boundary,
                building_iou=.5, lpips=.5, label_provenance='manually_reviewed')


def test_final_reproduction_allows_different_record_order(module):
    rows = [row('a', .2), row('b', .3)]
    images = {'a': 'hash-a', 'b': 'hash-b'}
    module.verify_final_reproduction(images, images, rows, list(reversed(rows)))


@pytest.mark.parametrize('corruption', ['image', 'metric', 'duplicate', 'dropped'])
def test_final_reproduction_rejects_changes(module, corruption):
    rows = [row('a', .2), row('b', .3)]
    new = copy.deepcopy(rows)
    images = {'a': 'hash-a', 'b': 'hash-b'}
    changed = dict(images)
    if corruption == 'image':
        changed['a'] = 'different'
    elif corruption == 'metric':
        new[0]['lpips'] = .1
    elif corruption == 'duplicate':
        new.append(copy.deepcopy(new[0]))
    else:
        new.pop()
    with pytest.raises(ResearchError):
        module.verify_final_reproduction(images, changed, rows, new)


def test_late_deterioration_keeps_negative_improvement(module):
    raw = {(n, s): [row('a', v), row('b', v)]
           for n in module.NAMES for s, v in ((300, .1), (400, .2), (500, .3))}
    result = module.late_changes(raw)
    comparison = result['A3']['300_to_500']['reviewed21']
    assert comparison['relative_boundary_reduction'] == pytest.approx(-2.)
    assert comparison['boundary_bootstrap']['mean_improvement'] == pytest.approx(-.2)
    assert comparison['boundary_bootstrap']['ci95'][1] < 0
