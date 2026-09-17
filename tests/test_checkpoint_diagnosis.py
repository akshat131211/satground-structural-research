import importlib.util
from pathlib import Path

import pytest

from satground.common import ResearchError


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    result = []
    for name in ('report_checkpoint_diagnosis', 'select_additional_review'):
        spec = importlib.util.spec_from_file_location(name, scripts / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def metric_row(sid, group, target, prediction, boundary, valid=1000):
    return dict(sample_id=sid, geo_group=group, target_building_pixels=target,
                predicted_building_pixels=prediction, valid_pixels=valid,
                label_provenance='independent_segmenter_pseudo_label',
                boundary_error=boundary, building_iou=0, lpips=.5, ssim=.2)


def test_tiny_false_positive_retains_penalty_and_reports_area(modules):
    report, _ = modules
    rows = [metric_row('empty', 'one', 0, 13, 1), metric_row('building', 'two', 200, 0, 1)]
    s = report.strata(rows)
    assert s['target_empty']['metrics']['boundary_error'] == 1
    assert s['target_empty']['group_weighted_false_positive_fraction'] == pytest.approx(.013)
    assert s['target_empty']['nonempty_predictions'] == 1
    assert s['target_building']['zero_building_predictions'] == 1
    assert s['target_building']['views'] == 1


def test_false_positive_area_weights_groups_not_images(modules):
    report, _ = modules
    rows = [metric_row('a', 'one', 0, 100, 1), metric_row('b', 'one', 0, 100, 1),
            metric_row('c', 'two', 0, 900, 1)]
    assert report.strata(rows)['target_empty']['group_weighted_false_positive_fraction'] == pytest.approx(.5)


def test_empty_stratum_is_unavailable_not_zero(modules):
    report, _ = modules
    s = report.strata([metric_row('a', 'one', 100, 0, 1)])
    assert s['target_empty']['views'] == 0
    assert s['target_empty']['metrics'] is None
    assert s['target_empty']['group_weighted_false_positive_fraction'] is None


def test_additional_review_excludes_accepted_scenes_and_rejects_audit(modules):
    _, selection = modules
    rows = [dict(sample_id=f'{i}-{yaw}', city='Chicago', split='validation', geo_group=str(i),
                 panorama=f'pano-{i}', satellite=f'tile-{i}') for i in range(5) for yaw in range(2)]
    chosen = selection.select(rows, {'0-0': {}}, 4)
    assert chosen == selection.select(list(reversed(rows)), {'0-0': {}}, 4)
    assert len({r['panorama'] for r in chosen}) == len({r['satellite'] for r in chosen}) == 4
    assert all(r['panorama'] != 'pano-0' for r in chosen)
    rows[-1]['split'] = 'audit'
    with pytest.raises(ResearchError, match='not allowed'):
        selection.select(rows, {'0-0': {}}, 4)
