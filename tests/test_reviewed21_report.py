import copy
import importlib

import pytest

from satground.common import ResearchError


@pytest.fixture
def check(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    return importlib.import_module('report_reviewed21').verify_label_only_change


def row(sid='new', provenance='independent_segmenter_pseudo_label'):
    return dict(sample_id=sid, geo_group='group', split='validation', lpips=.4,
                psnr=20., ssim=.6, entropy=.3, identical_rgb=False,
                label_provenance=provenance, boundary_error=.2, building_iou=.5,
                target_building_pixels=100, valid_pixels=200)


def test_only_new_reviewed_label_metrics_can_change_in_any_row_order(check):
    old = [row(), row('kept', 'manually_reviewed'), row('pseudo')]
    new = copy.deepcopy(old)
    new[0].update(label_provenance='manually_reviewed', boundary_error=.1,
                  building_iou=.7, target_building_pixels=120, valid_pixels=220)
    check(old, list(reversed(new)), {'new', 'kept'})


@pytest.mark.parametrize('key,value', [('lpips', .45), ('geo_group', 'different'),
                                      ('split', 'audit'), ('identical_rgb', True)])
def test_label_review_cannot_hide_nonlabel_drift(check, key, value):
    old = [row()]
    new = copy.deepcopy(old)
    new[0][key] = value
    with pytest.raises(ResearchError, match='Non-label'):
        check(old, new, {'new'})


@pytest.mark.parametrize('previously_reviewed', [True, False])
def test_preserved_targets_cannot_be_rescored_silently(check, previously_reviewed):
    old = [row(provenance='manually_reviewed' if previously_reviewed else 'pseudo')]
    new = copy.deepcopy(old)
    new[0]['boundary_error'] = .01
    with pytest.raises(ResearchError, match='unchanged target'):
        check(old, new, {'new'} if previously_reviewed else set())


@pytest.mark.parametrize('new', [[], [row(), row()], [row('replacement')]])
def test_dropped_duplicated_or_replaced_views_are_rejected(check, new):
    with pytest.raises(ResearchError, match='cohort changed'):
        check([row()], new, {'new'})
