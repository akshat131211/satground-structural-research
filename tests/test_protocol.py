import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from satground.adapter import StructuralAdapter
from satground.backend import load_style
from satground.camera import camera_matrices, crop_panorama
from satground.common import ROOT, ResearchError, read_jsonl, require_development, save_json, safe_data_path, write_jsonl
from satground.losses import structural_losses
from satground.manifests import assert_disjoint, prepare, require_files, validate_rgb
from satground.metrics import building_metrics, paired_group_bootstrap, server_gate
from satground.reliability import fit_calibration, reliability_metrics


def sample(split='train', sample_id='one', group='Chicago:0:0'):
    return dict(sample_id=sample_id, city='Chicago', split=split, geo_group=group,
                satellite='Chicago/satellite/a.png', panorama='Chicago/panorama/a.jpg')


def test_adapter_identity_and_frozen_renderer_gradients():
    torch.manual_seed(17)
    adapter = StructuralAdapter(channels=8, width=3)
    frozen_renderer = torch.nn.Conv2d(8, 3, 3, padding=1).eval().requires_grad_(False)
    features = torch.randn(1, 8, 12, 12)
    assert torch.equal(features, adapter(features))
    loss = frozen_renderer(adapter(features)).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None for p in frozen_renderer.parameters())


@pytest.mark.parametrize('yaw,pitch', [(0, 0), (90, 0), (-90, 15), (179, -20)])
def test_camera_matches_pinned_release(yaw, pitch):
    vendor = ROOT / 'external' / 'Sat3DGen'
    if not vendor.exists():
        pytest.skip('Run bootstrap for pinned-reference camera checks')
    sys.path.insert(0, str(vendor))
    from source.rendering.transform_perspective import compose_rotmat
    from source.rendering.pano2perspective import GetPerspective
    matrix, intrinsic, low = camera_matrices(20, -30, yaw, pitch, 90, 64)
    np.testing.assert_allclose(matrix[:3, :3], compose_rotmat(0, pitch, yaw), atol=1e-7)
    np.testing.assert_allclose(matrix[:3, 3], [-30 / 320, -20 / 320, -.85], atol=1e-7)
    np.testing.assert_allclose(low[:2], intrinsic[:2] / 2)
    y, x = np.mgrid[:128, :256]
    pano = np.stack([x, y * 2, (x + y) % 256], -1).astype(np.uint8)
    reference = GetPerspective(pano, [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]], yaw, pitch, 64, 64)
    ours = crop_panorama(pano, yaw, pitch, 90, 64)
    np.testing.assert_allclose(ours.astype(float), reference.astype(float), atol=2)


def test_positive_pitch_looks_down_in_panorama():
    image = np.tile(np.arange(128, dtype=np.uint8)[:, None, None], (1, 256, 3))
    up = crop_panorama(image, pitch=-20, size=32).mean()
    down = crop_panorama(image, pitch=20, size=32).mean()
    assert down > up


@pytest.mark.parametrize('key', ['satellite', 'panorama', 'geo_group'])
def test_shared_locations_never_cross_splits(key):
    first, second = sample(), sample('audit', 'two', 'Chicago:9:9')
    second.update(satellite='Chicago/satellite/b.png', panorama='Chicago/panorama/b.jpg')
    second[key] = first[key]
    with pytest.raises(ResearchError, match='leakage'):
        assert_disjoint([first, second])


def test_seattle_and_path_escape_rejected(tmp_path):
    row = sample()
    row['city'] = 'Seattle'
    with pytest.raises(ResearchError, match='sealed'):
        require_development([row])
    with pytest.raises(ResearchError, match='escapes'):
        safe_data_path(tmp_path, '../private.jpg')


def test_missing_rgb_reports_block_and_never_imputes(tmp_path):
    row = sample()
    write_jsonl(tmp_path / 'm.jsonl', [row])
    report = validate_rgb(tmp_path / 'm.jsonl', tmp_path / 'RGB', tmp_path / 'report.json')
    assert report['status'] == 'blocked' and report['missing_count'] == 2
    with pytest.raises(ResearchError, match='Original VIGOR'):
        require_files([row], tmp_path / 'RGB')


def test_geographic_manifest_reproducible(tmp_path):
    lines = []
    for i in range(150):
        lat, lon = 41.5 + i * .0027, -87.9 + (i % 7) * .013
        lines.append(f'Chicago/satellite/satellite_{lat}_{lon}.png Chicago/panorama/{i}.jpg Chicago/pano_sky_mask/{i}.png 0 0')
    source = tmp_path / 'pairs.txt'
    source.write_text('\n'.join(lines))
    counts = dict(train=10, validation=5, calibration=5, audit=5)
    prepare(source, tmp_path / 'first', counts)
    prepare(source, tmp_path / 'second', counts)
    assert (tmp_path / 'first/all.jsonl').read_bytes() == (tmp_path / 'second/all.jsonl').read_bytes()
    rows = read_jsonl(tmp_path / 'first/all.jsonl')
    assert_disjoint(rows)
    assert set(r['split'] for r in rows) == set(counts)
    assert all(not r['footprint_bound_verified'] for r in rows)


def test_only_real_training_style_allowed(tmp_path):
    path = tmp_path / 'style.json'
    for record in [dict(source_split='audit'), dict(source_split='train', synthetic=True)]:
        save_json(path, record)
        with pytest.raises(ResearchError):
            load_style(path)
    save_json(path, dict(source_split='train', synthetic=False, manifest_sha256='abc', histogram=[1 / 90] * 270))
    assert len(load_style(path, 'abc')) == 270
    with pytest.raises(ResearchError, match='different training manifest'):
        load_style(path, 'def')


def test_empty_masks_and_shifted_buildings():
    blank = np.zeros((32, 32), dtype=bool)
    building = blank.copy()
    building[8:20, 8:20] = True
    assert building_metrics(blank, blank)['building_iou'] == 1
    assert building_metrics(blank, building)['boundary_error'] == 1
    assert building_metrics(building, building)['boundary_error'] == 0
    shifted = np.roll(building, 3, axis=1)
    assert 0 < building_metrics(shifted, building)['boundary_error'] < 1
    with pytest.raises(ResearchError, match='no valid pixels'):
        building_metrics(building, building, blank)


def test_ignored_structural_pixels_have_no_gradient():
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    truth, ignored = torch.zeros_like(prediction), torch.zeros_like(prediction)
    region, boundary = structural_losses(prediction, truth, ignored)
    (region + boundary).backward()
    assert region == 0 and boundary == 0 and prediction.grad.abs().sum() == 0


def test_group_bootstrap_and_missing_pairs():
    a = [dict(sample_id=str(i), geo_group=str(i // 4), boundary_error=.2) for i in range(40)]
    b = [dict(r, boundary_error=.15) for r in a]
    ci = paired_group_bootstrap(a, b)
    assert ci['geographic_groups'] == 10
    assert ci['ci95'][0] > 0
    with pytest.raises(ResearchError, match='identical'):
        paired_group_bootstrap(a, b[:-1])


def test_server_gate_requires_all_evidence():
    assert server_gate([], {})['status'] == 'insufficient_evidence'
    runs = [dict(seed=s, relative_boundary_reduction=.15, boundary_ci95=[.01, .02], iou_change=0,
                 relative_lpips_change=0) for s in [17, 29, 43]]
    review = dict(human_review_complete=True, no_frequent_new_hallucinations=True,
                  provenance_complete=True, audit_frozen_before_evaluation=True)
    assert server_gate(runs, review)['status'] == 'pass'
    runs[0]['relative_lpips_change'] = .2
    assert server_gate(runs, review)['status'] == 'fail'


def test_calibration_rejects_overlap_and_single_outcome():
    rows = [dict(sample_id=str(i), geo_group=f'cal:{i // 5}', split='calibration', disagreement=i / 100,
                 boundary_error=.05 if i > 50 else .01) for i in range(100)]
    fit = fit_calibration(rows)
    with pytest.raises(ResearchError, match='overlaps'):
        reliability_metrics(rows, fit)
    heldout = [dict(r, sample_id='audit' + r['sample_id'], geo_group='audit' + r['geo_group'], split='audit') for r in rows]
    metrics = reliability_metrics(heldout, fit)
    assert metrics['brier'] < metrics['constant_brier']
    with pytest.raises(ResearchError, match='both success and failure'):
        fit_calibration([dict(r, boundary_error=0) for r in rows])


def test_checkpoint_resume_preserves_adapter_optimizer_and_rng(tmp_path):
    from satground.runs import atomic_checkpoint
    torch.manual_seed(12)
    model = StructuralAdapter(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    features = torch.randn(1, 4, 8, 8)
    def step():
        optimizer.zero_grad()
        loss = (model(features) - torch.rand_like(features)).square().mean()
        loss.backward()
        optimizer.step()
    step()
    state = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), rng=torch.get_rng_state())
    atomic_checkpoint(tmp_path / 'resume.pt', state)
    step()
    expected = copy.deepcopy(model.state_dict())
    saved = torch.load(tmp_path / 'resume.pt', weights_only=True)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng'])
    step()
    for key in expected:
        assert torch.equal(expected[key], model.state_dict()[key])


def test_selector_rejects_audit_metrics(tmp_path):
    from satground.evaluation import select_checkpoint
    save_json(tmp_path / 'summary.json', dict(group_weighted=dict(lpips=.2, boundary_error=.1), manifest_sha256='same'))
    write_jsonl(tmp_path / 'metrics.jsonl', [dict(split='audit')])
    save_json(tmp_path / 'candidates.json', dict(candidates=[dict(summary=str(tmp_path / 'summary.json'), checkpoint='fake.pt')]))
    with pytest.raises(ResearchError, match='only use validation'):
        select_checkpoint(tmp_path / 'candidates.json', tmp_path / 'summary.json', tmp_path / 'selection.json')


def test_readiness_rejects_main_training_without_real_review(tmp_path):
    from satground.readiness import require_training_readiness
    from satground.common import sha256
    row = sample()
    manifest = tmp_path / 'all.jsonl'
    report_path, review_path = tmp_path / 'data.json', tmp_path / 'qa.json'
    write_jsonl(manifest, [row])
    save_json(report_path, dict(status='ready_for_review', missing_count=0, invalid=[], cross_split_duplicate_hashes=[],
                               manifest_sha256=sha256(manifest)))
    save_json(review_path, dict(footprint_bound_verified=False))
    config = dict(data_validation=str(report_path), data_qa=str(review_path), all_manifest=str(manifest))
    assert require_training_readiness(config, [row], 100)['stage'] == 'preliminary_learning_check'
    with pytest.raises(ResearchError, match='documented footprint'):
        require_training_readiness(config, [row], 101)


def test_bad_histogram_cannot_condition_a_scientific_run(tmp_path):
    save_json(tmp_path / 'bad.json', dict(source_split='train', synthetic=False, histogram=[0] * 270))
    with pytest.raises(ResearchError, match='sum to one'):
        load_style(tmp_path / 'bad.json')


def test_position_stress_keeps_full_magnitude_at_tile_edges():
    from satground.camera import perturb_position_width
    for original in [-320, -315, 0, 310, 315, 320]:
        perturbed = perturb_position_width(original)
        assert -320 <= perturbed <= 320
        assert abs(perturbed - original) == 10
