"""CPU-only controlled geometry tests; no imagery, model loading or updates."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from valid_surface_objective import (
    COEFFICIENT, MASS_FLOOR, POLICY, RADIUS, _distances,
    prepare_surface_labels, surface_support_counts, valid_surface_loss,
)


def scene():
    y = np.zeros((64, 64), bool)
    y[18:46, 18:46] = True
    return y, np.ones_like(y)


def tensor(mask):
    return torch.from_numpy(np.where(mask, 0.99, 0.01).astype(np.float32))


def full_structural(probability, y, v, prepared):
    valid = torch.from_numpy(v.astype(np.float32))
    target = torch.from_numpy(y.astype(np.float32))
    region = (F.binary_cross_entropy(probability, target, reduction='none') * valid).sum() / valid.sum().clamp_min(1)
    return 0.5 * region + COEFFICIENT * valid_surface_loss(probability, prepared)


def test_fixed_policy_and_input_preservation():
    y, v = scene()
    yc, vc = y.copy(), v.copy()
    prepared = prepare_surface_labels(y, v)
    assert RADIUS == 8 and MASS_FLOOR == 32 and COEFFICIENT == 0.5
    assert prepared['provenance']['policy'] == POLICY
    assert np.array_equal(y, yc) and np.array_equal(v, vc)
    assert not prepared['weights'].flags.writeable
    assert prepared['counts']['historical_contour_pixels_not_sources'] >= 0
    assert surface_support_counts(y, v)['interface_source_pixels'] == int(prepared['interface'].sum())


def test_correct_prediction_and_controlled_displacements():
    y, v = scene()
    prepared = prepare_surface_labels(y, v)
    losses = [float(valid_surface_loss(tensor(np.roll(y, shift, axis=1)), prepared)) for shift in (0, 1, 2, 4, 8)]
    assert 0 <= losses[0] < losses[1] < losses[2] < losses[3] < losses[4]
    assert float(valid_surface_loss(torch.from_numpy(y.astype(np.float32)), prepared)) < 2e-6


@pytest.mark.parametrize('failure', ['missing', 'extra', 'inverted', 'empty', 'full', 'flat'])
def test_incorrect_structure_does_not_beat_correct(failure):
    y, v = scene()
    prepared = prepare_surface_labels(y, v)
    wrong = y.copy()
    if failure in ('missing', 'empty'):
        wrong[:] = False
    elif failure == 'extra':
        wrong[5:13, 5:13] = True  # Mostly outside the surface band: region still penalizes it.
    elif failure == 'inverted':
        wrong = ~y
    elif failure == 'full':
        wrong[:] = True
    p = torch.full(y.shape, 0.5) if failure == 'flat' else tensor(wrong)
    assert full_structural(p, y, v, prepared) > full_structural(tensor(y), y, v, prepared)


def test_probability_and_logit_gradients_restore_missing_and_extra_pixels():
    y, v = scene()
    prepared = prepare_surface_labels(y, v)
    p = torch.full(y.shape, 0.5, requires_grad=True)
    valid_surface_loss(p, prepared).backward()
    w = prepared['weights']
    assert torch.all(p.grad[torch.from_numpy((w[1] > 0).copy())] < 0)
    assert torch.all(p.grad[torch.from_numpy((w[0] > 0).copy())] > 0)
    assert torch.all(p.grad[torch.from_numpy((w.sum(axis=0) == 0).copy())] == 0)
    z = torch.zeros(y.shape, requires_grad=True)
    valid_surface_loss(z.sigmoid(), prepared).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert torch.all(z.grad[torch.from_numpy((w[1] > 0).copy())] < 0)
    assert torch.all(z.grad[torch.from_numpy((w[0] > 0).copy())] > 0)


def test_unknown_changes_cannot_change_weights_loss_or_scored_gradients():
    y, v = scene()
    v[25:38, 28:37] = False
    changed = y.copy()
    changed[~v] = ~changed[~v]
    a, b = prepare_surface_labels(y, v), prepare_surface_labels(changed, v)
    assert np.array_equal(a['weights'], b['weights'])
    p = torch.full(y.shape, 0.4, requires_grad=True)
    q = p.detach().clone()
    q[torch.from_numpy(~v)] = 0.9
    q.requires_grad_(True)
    la, lb = valid_surface_loss(p, a), valid_surface_loss(q, b)
    la.backward(); lb.backward()
    assert float(la) == float(lb)
    assert torch.equal(p.grad, q.grad)
    assert torch.all(p.grad[torch.from_numpy(~v)] == 0)


def test_invalid_strip_blocks_distance_to_an_unseeded_component():
    y = np.zeros((48, 48), bool)
    y[10:20, 5:15] = True
    v = np.ones_like(y)
    v[:, 23:26] = False
    prepared = prepare_surface_labels(y, v)
    assert prepared['counts']['domain_components_without_observed_interface'] == 1
    assert np.isinf(prepared['distance'][:, 27:]).all()
    assert not prepared['weights'][:, :, 27:].any()


def test_shortest_paths_have_euclidean_diagonals_but_cannot_cut_invalid_corners():
    domain = np.zeros((7, 7), bool)
    domain[1:6, 1:6] = True
    seeds = np.zeros_like(domain); seeds[2, 2] = True
    assert _distances(domain, seeds)[3, 3] == pytest.approx(np.sqrt(2))
    domain[2, 3] = False
    assert _distances(domain, seeds)[3, 3] == pytest.approx(2)
    domain[3, 2] = False
    assert _distances(domain, seeds)[3, 3] > 2


def test_crop_border_never_closes_a_full_building_mask():
    y = np.ones((32, 32), bool)
    prepared = prepare_surface_labels(y, np.ones_like(y))
    assert not prepared['interface'].any()
    assert not prepared['weights'].any()
    assert not prepared['domain'][[0, -1], :].any()
    assert not prepared['domain'][:, [0, -1]].any()


@pytest.mark.parametrize('building,valid', [(False, True), (True, True), (False, False), (True, False)])
def test_empty_support_is_finite_zero_with_a_live_zero_gradient(building, valid):
    y, v = np.full((12, 12), building), np.full((12, 12), valid)
    prepared = prepare_surface_labels(y, v)
    p = torch.full((1, 1, 12, 12), 0.5, requires_grad=True)
    loss = valid_surface_loss(p, prepared)
    loss.backward()
    assert float(loss) == 0 and torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0
    assert np.all(prepared['denominators'] == MASS_FLOOR)


def test_tiny_fragment_cannot_acquire_unbounded_inverse_mass_weight():
    y = np.zeros((32, 32), bool); y[16, 16] = True
    prepared = prepare_surface_labels(y, np.ones_like(y))
    assert prepared['counts']['class_weight_mass'][1] < MASS_FLOOR
    p = torch.full(y.shape, 0.5, requires_grad=True)
    valid_surface_loss(p, prepared).backward()
    assert p.grad.abs().max() <= 0.5 / MASS_FLOOR
    assert abs(float(p.grad[16, 16])) == pytest.approx(0.5 / (9 * MASS_FLOOR))


def test_valid_neighbour_rule_exposes_removed_historical_support():
    y = np.zeros((12, 12), bool); y[1:, :] = True
    counts = surface_support_counts(y, np.ones_like(y))
    assert counts['historical_positive_contour_pixels'] > 0
    assert counts['interface_source_pixels'] == 0
    assert counts['historical_contour_pixels_not_sources'] == counts['historical_positive_contour_pixels']


def test_float32_runtime_retains_differentiability_from_float64_input():
    y, v = scene()
    p = torch.full(y.shape, 0.5, dtype=torch.float64, requires_grad=True)
    loss = valid_surface_loss(p, prepare_surface_labels(y, v))
    loss.backward()
    assert loss.dtype == torch.float32 and p.grad.dtype == torch.float64 and torch.isfinite(p.grad).all()


@pytest.mark.parametrize('failure', ['binary', 'shape', 'nan', 'integer', 'batch'])
def test_malformed_inputs_fail_closed(failure):
    y, v = scene()
    if failure == 'binary':
        bad = y.astype(float); bad[0, 0] = np.nan
        with pytest.raises(ValueError): prepare_surface_labels(bad, v)
        return
    if failure == 'shape':
        with pytest.raises(ValueError): prepare_surface_labels(y, v[:-1])
        return
    p = torch.full(y.shape, 0.5)
    if failure == 'nan': p[0, 0] = float('nan')
    elif failure == 'integer': p = p.long()
    else: p = p.repeat(2, 1, 1)
    with pytest.raises(ValueError): valid_surface_loss(p, prepare_surface_labels(y, v))
