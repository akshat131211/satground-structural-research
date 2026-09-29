"""CPU regressions for the fixed saturation correction, preserving failed v1."""
from pathlib import Path
import heapq
import math
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import saturated_surface_objective as v2
from valid_surface_objective import (
    prepare_surface_labels as prepare_v1,
    valid_surface_loss as loss_v1,
)
from compare_saturated_surface_cases import (
    PRIOR_GATE, build_report, measure_candidate, write_comparison,
)
from contour_objective_cases import fixed_cases, probability_from_mask, rectangle
from satground.common import ResearchError, load_json


def scene(size=32):
    y = np.zeros((size, size), bool)
    y[8:24, 8:24] = True
    return y, np.ones_like(y)


def _full_distances(domain, seeds):
    """Independent uncapped shortest-path oracle for small CPU fixtures."""
    distances = np.full(domain.shape, np.inf, np.float64)
    distances[seeds] = 0
    heap = [(0., int(r), int(c)) for r, c in np.argwhere(seeds)]
    heapq.heapify(heap)
    neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    while heap:
        value, r, c = heapq.heappop(heap)
        if value != distances[r, c]:
            continue
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < domain.shape[0] and 0 <= cc < domain.shape[1] and domain[rr, cc]):
                continue
            if dr and dc and not (domain[r, cc] and domain[rr, c]):
                continue
            next_value = value + (math.sqrt(2) if dr and dc else 1.)
            if next_value < distances[rr, cc]:
                distances[rr, cc] = next_value
                heapq.heappush(heap, (next_value, rr, cc))
    return distances


@pytest.fixture(scope='module')
def report():
    return build_report()


@pytest.fixture(scope='module')
def rows(report):
    return {row['name']: row for row in report['cases']}


def test_fixed_formula_and_counts_are_not_relabelled_as_band():
    y, valid = scene()
    prepared = v2.prepare_surface_labels(y, valid)
    assert v2.POLICY == 'valid_surface_saturated_v2'
    assert (v2.RADIUS, v2.MASS_FLOOR, v2.COEFFICIENT) == (8, 32, .5)
    counts = prepared['counts']
    assert 'band_pixels' not in counts and 'class_band_pixels' not in counts
    assert counts['surface_scored_pixels'] == int(prepared['reachable'].sum())
    assert counts['near_band_pixels'] <= counts['surface_scored_pixels']
    assert counts['saturated_distance_pixels'] == int((prepared['reachable'] & (prepared['distance'] >= 8)).sum())
    assert counts['beyond_radius_saturated_pixels'] == counts['surface_scored_pixels'] - counts['near_band_pixels']
    assert sum(counts['class_scored_pixels']) == counts['surface_scored_pixels']
    assert counts['class_weight_mass'] == prepared['weights'].sum(axis=(1, 2), dtype=np.float64).tolist()
    assert counts['class_denominators'] == prepared['denominators'].tolist()
    assert np.all(prepared['denominators'] >= 32)
    reachable = prepared['reachable']
    np.testing.assert_allclose(prepared['weights'].sum(axis=0)[reachable], (1 + prepared['distance'][reachable]) / 9, rtol=1e-7)
    assert np.all(prepared['weights'].sum(axis=0)[~reachable] == 0)


def test_reuses_exact_old_sources_and_near_distances_without_mutating_inputs():
    y, valid = scene()
    copies = [array.copy() for array in (y, valid)]
    old = prepare_v1(y, valid)
    before = {key: value.copy() for key, value in old.items() if isinstance(value, np.ndarray)}
    new = v2.prepare_surface_labels(y, valid)
    assert np.array_equal(y, copies[0]) and np.array_equal(valid, copies[1])
    assert np.array_equal(old['domain'], new['domain'])
    assert np.array_equal(old['interface'], new['interface'])
    near = np.isfinite(old['distance'])
    assert np.array_equal(old['distance'][near], new['distance'][near])
    for key, value in before.items():
        assert np.array_equal(value, old[key])
        assert not old[key].flags.writeable
    assert all(not value.flags.writeable for value in new.values() if isinstance(value, np.ndarray))
    for key in ('historical_positive_contour_pixels', 'interface_source_pixels', 'historical_contour_pixels_not_sources'):
        assert new['counts'][key] == old['counts'][key]


def test_capped_distance_matches_independent_full_geodesics_with_obstacles():
    y = np.zeros((28, 28), bool)
    y[6:14, 4:11] = True
    valid = np.ones_like(y)
    valid[4:22, 15:18] = False
    valid[18:21, 6:10] = False
    prepared = v2.prepare_surface_labels(y, valid)
    actual = _full_distances(prepared['domain'], prepared['interface'])
    expected = np.minimum(actual, 8)
    expected[~np.isfinite(actual)] = np.inf
    np.testing.assert_allclose(prepared['distance'], expected, rtol=0, atol=1e-12)
    assert np.array_equal(prepared['reachable'], np.isfinite(actual))


def test_valid_barrier_does_not_fill_an_unseeded_component():
    y = np.zeros((40, 40), bool)
    y[8:16, 4:12] = True
    valid = np.ones_like(y)
    valid[:, 19:22] = False
    prepared = v2.prepare_surface_labels(y, valid)
    assert prepared['counts']['unsupported_components'] == 1
    assert prepared['counts']['domain_components_without_observed_interface'] == 1
    assert not prepared['reachable'][:, 23:].any()
    assert not prepared['weights'][:, :, 23:].any()
    assert np.isinf(prepared['distance'][:, 23:]).all()
    assert not prepared['weights'][:, ~prepared['domain']].any()


def test_corner_touching_valid_components_never_share_interface_support():
    y = np.zeros((20, 20), bool)
    valid = np.zeros_like(y)
    valid[1:9, 1:9] = True
    valid[9:18, 9:18] = True
    y[3:6, 3:6] = True
    prepared = v2.prepare_surface_labels(y, valid)
    assert prepared['counts']['domain_components'] == 2
    assert prepared['counts']['unsupported_components'] == 1
    assert not prepared['reachable'][10:17, 10:17].any()
    assert np.isinf(prepared['distance'][10:17, 10:17]).all()


def test_unknown_target_and_probability_changes_do_not_change_weights_or_gradients():
    y, valid = scene()
    valid[12:20, 13:19] = False
    changed = y.copy()
    changed[~valid] = ~changed[~valid]
    a, b = v2.prepare_surface_labels(y, valid), v2.prepare_surface_labels(changed, valid)
    assert np.array_equal(a['weights'], b['weights'])
    assert np.array_equal(a['denominators'], b['denominators'])
    p = torch.full(y.shape, .4, requires_grad=True)
    q = p.detach().clone()
    q[torch.from_numpy(~valid)] = .95
    q.requires_grad_(True)
    la, lb = v2.valid_surface_loss(p, a), v2.valid_surface_loss(q, b)
    ga, gb = torch.autograd.grad(la, p)[0], torch.autograd.grad(lb, q)[0]
    assert float(la) == float(lb)
    assert torch.equal(ga, gb)
    assert torch.all(ga[torch.from_numpy(~valid)] == 0)


@pytest.mark.parametrize('building,valid', [(True, True), (False, True), (True, False), (False, False)])
def test_no_interface_support_is_finite_zero_without_invented_crop_edges(building, valid):
    y = np.full((16, 16), building)
    prepared = v2.prepare_surface_labels(y, np.full_like(y, valid))
    assert not prepared['interface'].any() and not prepared['weights'].any()
    assert prepared['counts']['surface_scored_pixels'] == 0
    assert prepared['counts']['near_band_pixels'] == 0
    assert not prepared['domain'][[0, -1], :].any()
    p = torch.full(y.shape, .5, requires_grad=True)
    loss = v2.valid_surface_loss(p, prepared)
    gradient = torch.autograd.grad(loss, p)[0]
    assert float(loss) == 0 and torch.isfinite(gradient).all() and gradient.abs().sum() == 0
    assert np.all(prepared['denominators'] == 32)


def test_probability_gradients_correct_all_seeded_support_including_far_pixels():
    y, valid = scene()
    prepared = v2.prepare_surface_labels(y, valid)
    p = torch.full(y.shape, .5, dtype=torch.float64, requires_grad=True)
    loss = v2.valid_surface_loss(p, prepared)
    gradient = torch.autograd.grad(loss, p)[0]
    assert loss.dtype == torch.float32 and gradient.dtype == torch.float64
    assert torch.isfinite(gradient).all()
    foreground = torch.from_numpy((prepared['weights'][1] > 0).copy())
    background = torch.from_numpy((prepared['weights'][0] > 0).copy())
    none = ~(foreground | background)
    assert torch.all(gradient[foreground] < 0) and torch.all(gradient[background] > 0)
    assert torch.all(gradient[none] == 0)
    beyond = torch.from_numpy((prepared['reachable'] & (prepared['distance'] == 8)).copy())
    assert beyond.any() and torch.all(gradient[beyond].abs() > 0)


def test_mass_floor_keeps_tiny_foreground_gradients_bounded():
    y = np.zeros((24, 24), bool)
    y[12, 12] = True
    prepared = v2.prepare_surface_labels(y, np.ones_like(y))
    assert prepared['counts']['class_weight_mass'][1] < 32
    assert prepared['denominators'][1] == 32
    p = torch.full(y.shape, .5, requires_grad=True)
    gradient = torch.autograd.grad(v2.valid_surface_loss(p, prepared), p)[0]
    assert gradient.abs().max() <= .5 / 32
    assert float(gradient[12, 12]) == pytest.approx(-.5 / (9 * 32))


def test_old_far_reversal_is_preserved_and_actual_new_term_removes_it():
    cases = {case['name']: case for case in fixed_cases()}
    target = cases['hard_translation_dx40']['target']
    valid = cases['hard_translation_dx40']['valid']
    old_prepared = prepare_v1(target.numpy(), valid.numpy())
    new_prepared = v2.prepare_surface_labels(target.numpy(), valid.numpy())
    p40, p44 = [cases[f'hard_translation_dx{dx}']['probability'] for dx in (40, 44)]
    old40, old44 = float(loss_v1(p40, old_prepared)), float(loss_v1(p44, old_prepared))
    new40, new44 = float(v2.valid_surface_loss(p40, new_prepared)), float(v2.valid_surface_loss(p44, new_prepared))
    assert old40 == pytest.approx(.5187076926231384) and old44 == pytest.approx(.4999999701976776)
    assert old44 < old40  # No v1 repair, replacement or relabelling.
    assert new44 >= new40
    assert new_prepared['weights'][0, 60, 110] == 1
    assert old_prepared['weights'][0, 60, 110] == 0


def test_report_retains_52_cases_and_historical_losses_exactly(report, rows):
    assert report['fixed_case_identity_sha256'] == '724d8fb9e94ef38674c1baa22a328f78e344a8dc477ab7adc36a6f2876edac77'
    assert report['fixed_synthetic_case_count'] == 52 and report['case_order_unchanged']
    assert [row['name'] for row in report['cases']] == [case['name'] for case in fixed_cases()]
    assert report['historical_losses_reproduce_exactly']
    for row in rows.values():
        assert row['historical_region'] == row['historical_losses']['region']
        assert row['historical_boundary'] == row['historical_losses']['boundary']
        assert row['candidate_losses']['combined_region_surface'] == pytest.approx(
            .5 * row['historical_region'] + .5 * row['candidate_losses']['unweighted_surface'])


def test_original_near_and_complete_translation_requirements_are_explicit(report, rows):
    assert PRIOR_GATE['near_shift_pixels'] == [0, 1, 2, 4, 8]
    assert PRIOR_GATE['complete_translation_pixels'] == [0, 1, 2, 4, 8, 16, 24, 32, 40, 44]
    assert all(report['near_shift_monotonicity'].values())
    assert all(report['complete_hard_translation_monotonicity'].values())
    family = [rows[f'hard_translation_dx{dx}'] for dx in PRIOR_GATE['complete_translation_pixels']]
    for name in ('unweighted_surface', 'combined_region_surface'):
        assert all(a['candidate_losses'][name] <= b['candidate_losses'][name] for a, b in zip(family, family[1:]))
    far = report['far_displacement_limitations']
    assert not far['combined_decreases_with_larger_far_displacement']
    assert not report['harmful_far_combined_ordering']
    assert report['controlled_gate_passed'] and report['training_clearance']
    assert report['training_clearance_scope'].startswith('Controlled-case prerequisite only')
    assert not report['generator_improvement_established'] and not report['server_gate_eligible']


def test_inversion_missing_extra_collapse_and_ignored_gates_remain_required(report):
    assert all(report['incorrect_case_checks'].values())
    assert 'inverted_reference' in report['incorrect_case_checks']
    for name in ('uniform_probability_0', 'uniform_probability_0.05', 'uniform_probability_0.5',
                 'uniform_probability_0.95', 'uniform_probability_1', 'extra_building_x110'):
        assert name in report['incorrect_case_checks']
    assert all(all(check['exact_invariance'].values()) for check in report['ignored_region_checks'])
    assert all(check['finite'] and check['correct_observed_support_directions'] for check in report['gradient_checks']['interior_uniform_checks'])
    assert report['gradient_checks']['gradient_continuity_at_half']
    assert all(row['gradient_l1'] == 0 for row in report['gradient_checks']['saturated_endpoints'])
    assert not report['real_data_read'] and not report['gpu_used'] and report['optimizer_updates'] == 0


@pytest.mark.parametrize('failure', ['nonbinary', 'shape', 'probability', 'nan', 'wrong_policy'])
def test_bad_masks_probabilities_and_old_policy_fail_closed(failure):
    y, valid = scene()
    if failure == 'nonbinary':
        bad = y.astype(float); bad[0, 0] = .5
        with pytest.raises(ValueError): v2.prepare_surface_labels(bad, valid)
        return
    if failure == 'shape':
        with pytest.raises(ValueError): v2.prepare_surface_labels(y, valid[:-1])
        return
    p = torch.full(y.shape, .5)
    if failure == 'probability': p[0, 0] = 1.2
    elif failure == 'nan': p[0, 0] = float('nan')
    prepared = prepare_v1(y, valid) if failure == 'wrong_policy' else v2.prepare_surface_labels(y, valid)
    with pytest.raises(ValueError): v2.valid_surface_loss(p, prepared)


def test_cpu_comparison_freshness_and_nonfinite_output_fail_closed(tmp_path, monkeypatch, report):
    import compare_saturated_surface_cases as comparison
    monkeypatch.setattr(comparison, 'build_comparison', lambda: report)
    output = tmp_path / 'result.json'
    assert write_comparison(output) == load_json(output)
    with pytest.raises(ResearchError, match='fresh'):
        write_comparison(output)
    (tmp_path / 'partial.json.tmp').write_text('partial', encoding='utf-8')
    with pytest.raises(ResearchError, match='Partial'):
        write_comparison(tmp_path / 'partial.json')
    monkeypatch.setattr(comparison, 'build_comparison', lambda: {'bad': float('nan')})
    with pytest.raises(ResearchError, match='nonfinite'):
        write_comparison(tmp_path / 'nonfinite.json')
    assert not (tmp_path / 'nonfinite.json').exists()
    assert 'sample_id' not in output.read_text() and 'reviewer' not in output.read_text()
