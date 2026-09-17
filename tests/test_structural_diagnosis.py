import importlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from satground.common import ResearchError, save_json, sha256, write_jsonl


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    return importlib.import_module('diagnose_objective_gradients'), importlib.import_module('prepare_pairing_context')


def test_gradients_preserve_magnitudes_and_conflicting_directions(modules):
    grad, _ = modules
    result = grad.vector_summary(dict(rgb=torch.tensor([1., 0.]), lpips=torch.tensor([0., 1.]),
                                     region=torch.tensor([2., 0.]), boundary=torch.tensor([-1., 0.])))
    assert result['boundary_norm_ratios']['region'] == .5
    assert result['cosines']['boundary_vs_region'] == -1
    assert result['weighted_norms']['A2'] == pytest.approx(np.sqrt(10))
    assert result['weighted_norms']['A3'] == pytest.approx(np.sqrt(5))


def test_zero_gradients_do_not_manufacture_ratios_or_agreement(modules):
    grad, _ = modules
    result = grad.vector_summary({k: torch.zeros(3) for k in grad.TERMS})
    assert all(v == 0 for v in result['weighted_norms'].values())
    assert all(v is None for v in result['boundary_norm_ratios'].values())
    assert all(v is None for v in result['cosines'].values())


def test_nonfinite_component_is_rejected(modules):
    grad, _ = modules
    vectors = {k: torch.zeros(3) for k in grad.TERMS}
    vectors['boundary'][0] = float('nan')
    with pytest.raises(ResearchError, match='Non-finite'):
        grad.vector_summary(vectors)


def test_gradient_check_rejects_material_decomposition_error(modules):
    grad, _ = modules
    assert grad.verify_gradient_sum(torch.ones(3), torch.ones(3)) == 0
    with pytest.raises(ResearchError, match='differs'):
        grad.verify_gradient_sum(torch.ones(3), torch.ones(3) * 1.01)


def test_gradient_report_rejects_partial_or_changed_cohorts(modules):
    import copy
    reporting = importlib.import_module('report_structural_gradients')
    rows = [dict(sample_id='a', city='Chicago', split='train', geo_group='g', checkpoint_step=step,
        valid_pixels=100, interior_valid_pixels=80, scored_building_pixels=20, target_boundary_pixels=10,
        image_gradient_sum_relative_error=1e-6, adapter_gradient_sum_relative_error=1e-6)
        for step in reporting.STEPS]
    assert len(reporting.validate_records(rows, expected_views=1)) == 3
    with pytest.raises(ResearchError, match='Incomplete'):
        reporting.validate_records(rows[:-1], expected_views=1)
    changed = copy.deepcopy(rows); changed[-1]['valid_pixels'] += 1
    with pytest.raises(ResearchError, match='labels changed'):
        reporting.validate_records(changed, expected_views=1)
    changed = copy.deepcopy(rows); changed[-1]['split'] = 'validation'
    with pytest.raises(ResearchError, match='not allowed'):
        reporting.validate_records(changed, expected_views=1)


def test_gradient_distribution_weights_groups_and_preserves_undefined(modules):
    reporting = importlib.import_module('report_structural_gradients')
    rows = [dict(geo_group='a', value=1), dict(geo_group='a', value=1),
            dict(geo_group='b', value=9), dict(geo_group='b', value=None)]
    summary = reporting.distribution(rows, lambda r: r['value'])
    assert summary['median'] == 1
    assert summary['equal_group_mean'] == 5
    assert summary['available'] == 3 and summary['unavailable'] == 1


def test_uncertain_edge_band_removes_positive_boundary_signal(modules):
    audit = importlib.import_module('audit_boundary_support')
    from satground.losses import soft_boundary, structural_losses
    building = np.zeros((16, 16), np.float32); building[:, 8:] = 1
    valid = np.ones_like(building); valid[:, 7:9] = 0
    counts = audit.support_counts(building, valid)
    assert counts['raw_target_boundary_pixels'] > 0
    assert counts['scored_building_pixels'] > 0
    assert counts['after_interior_validity_pixels'] == 0
    assert counts['labelled_building_without_positive_boundary']
    prediction = torch.linspace(.1, .9, 256).reshape(1, 1, 16, 16).requires_grad_()
    v = torch.from_numpy(valid)[None, None]
    _, boundary = structural_losses(prediction, torch.from_numpy(building)[None, None], v)
    interior = -torch.nn.functional.max_pool2d(-v, 3, stride=1, padding=1)
    smoothing = (soft_boundary(prediction) * interior).sum() / interior.sum()
    assert torch.allclose(boundary, smoothing)


def test_boundary_audit_distinguishes_empty_targets_from_lost_edges(modules):
    audit = importlib.import_module('audit_boundary_support')
    empty = audit.support_counts(np.zeros((8, 8), bool), np.ones((8, 8), bool))
    assert not empty['labelled_building_without_positive_boundary']
    building = np.zeros((8, 8), bool); building[:, 4:] = True
    scored = audit.support_counts(building, np.ones_like(building))
    assert scored['after_interior_validity_pixels'] > 0
    assert not scored['labelled_building_without_positive_boundary']


@pytest.mark.parametrize('yaw', [90., 180., -90.])
def test_context_uses_actual_nonzero_target_yaw(modules, tmp_path, monkeypatch, yaw):
    _, context = modules
    monkeypatch.setattr(context, 'check_vendor', lambda: tmp_path)
    monkeypatch.setattr(context, 'safe_data_path', lambda root, relative: tmp_path / relative)
    pano = np.zeros((256, 1024, 3), np.uint8)
    pano[:, :, 0] = np.arange(1024)[None, :] % 256
    pano[:, :, 1] = np.arange(256)[:, None]
    Image.fromarray(pano).save(tmp_path / 'panorama.png')
    Image.new('RGB', (640, 640)).save(tmp_path / 'satellite.png')
    row = dict(sample_id='synthetic', city='Chicago', geo_group='fixture', split='validation',
               satellite='satellite.png', panorama='panorama.png', h_offset=0, w_offset=0,
               yaw=yaw, pitch=0., fov=90., size=256)
    entry = dict(reviewed=True, reviewer='synthetic_fixture', reviewed_utc='2026-09-17T00:00:00+00:00', review_view_number=1)
    arrays = dict(target_image=context.crop_panorama(pano, yaw, 0, 90, 256),
                  building_mask=np.zeros((256, 256), np.uint8), valid_mask=np.full((256, 256), 255, np.uint8))
    for key, array in arrays.items():
        path = tmp_path / (key + '.png'); Image.fromarray(array).save(path)
        entry[key] = str(path); entry[key + '_sha256'] = sha256(path)
    save_json(tmp_path / 'index.json', {'synthetic': entry})
    write_jsonl(tmp_path / 'manifest.jsonl', [row])
    result = context.prepare(tmp_path / 'index.json', tmp_path / 'manifest.jsonl', tmp_path / 'context')
    view = result['views'][0]
    assert view['target_yaw'] == yaw
    first = tmp_path / 'context' / f"view1-yaw{view['rays'][0]['yaw']}.png"
    np.testing.assert_array_equal(np.asarray(Image.open(first)), arrays['target_image'])
    expected = {90.: [1, 0], 180.: [0, 1], -90.: [-1, 0]}[yaw]
    np.testing.assert_allclose(view['rays'][0]['center_ray_direction'], expected, atol=1e-7)
    assert not result['real_world_pairing_verified']
