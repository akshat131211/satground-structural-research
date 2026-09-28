"""Screen one fixed candidate on the prior synthetic cases, without training.

The tests preserve documented far-band and saturation limitations instead of
altering the cases or objective to conceal them.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from compare_surface_controlled_cases import (
    PRIOR_GATE, build_comparison, candidate_gradient_checks, candidate_parameter_probe,
    measure_candidate, write_comparison,
)
from contour_objective_cases import fixed_cases, probability_from_mask, rectangle
from satground.common import ResearchError, load_json


@pytest.fixture(scope='module')
def comparison():
    return build_comparison()


@pytest.fixture(scope='module')
def rows(comparison):
    return {row['name']: row for row in comparison['cases']}


def test_prior_gate_and_frozen_cases_are_preserved(comparison):
    assert comparison['fixed_synthetic_case_count'] == 52
    assert comparison['fixed_case_identity_sha256'] == '724d8fb9e94ef38674c1baa22a328f78e344a8dc477ab7adc36a6f2876edac77'
    assert comparison['case_order_unchanged'] and comparison['historical_losses_reproduce_exactly']
    assert [row['name'] for row in comparison['cases']] == [case['name'] for case in fixed_cases()]
    assert comparison['prior_gate']['radius'] == 8
    assert comparison['prior_gate']['mass_floor'] == 32
    assert comparison['prior_gate']['surface_coefficient'] == .5
    assert comparison['prior_gate']['region_coefficient'] == .5
    assert not comparison['real_data_read'] and not comparison['gpu_used']
    assert comparison['optimizer_updates'] == 0
    assert not comparison['generator_improvement_established'] and not comparison['server_gate_eligible']


def test_historical_region_and_boundary_are_exactly_retained(rows):
    for row in rows.values():
        assert row['historical_region'] == row['historical_losses']['region']
        assert row['historical_boundary'] == row['historical_losses']['boundary']
        values = row['candidate_losses']
        assert values['weighted_surface'] == pytest.approx(.5 * values['unweighted_surface'])
        assert values['combined_region_surface'] == pytest.approx(.5 * row['historical_region'] + .5 * values['unweighted_surface'])


def test_declared_near_shift_gate_passes_without_changing_cases(comparison, rows):
    assert comparison['near_shift_monotonicity'] == dict(unweighted_surface=True, combined_region_surface=True)
    keys = [f'hard_translation_dx{shift}' for shift in (0, 1, 2, 4, 8)]
    for name in ('unweighted_surface', 'combined_region_surface'):
        assert all(rows[a]['candidate_losses'][name] < rows[b]['candidate_losses'][name] for a, b in zip(keys, keys[1:]))


def test_incorrect_combined_structure_above_matched_is_explicit(comparison, rows):
    assert comparison['incorrect_case_checks']
    assert all(comparison['incorrect_case_checks'].values())
    reference = rows['hard_translation_dx0']['candidate_losses']['combined_region_surface']
    inverse = comparison['inversion_supplemental_check']
    assert inverse['candidate_losses']['combined_region_surface'] > reference
    assert 'inverted_reference' not in rows  # Supplement, not one of the original52.
    for name in PRIOR_GATE['incorrect_combined_cases']:
        if name != 'inverted_reference':
            assert rows[name]['candidate_losses']['combined_region_surface'] > reference
    # Exact endpoint collapse is also in the unchanged case set; clipping keeps
    # its gradient zero but its incorrect combined loss remains above matched.
    for name in ('uniform_probability_0', 'uniform_probability_1'):
        assert rows[name]['candidate_losses']['combined_region_surface'] > reference


def test_missing_uniform_prediction_now_has_live_directional_surface_gradient(comparison, rows):
    missing = rows['uniform_probability_0.05']
    assert missing['historical_probability_gradients']['boundary']['l1'] == 0
    assert missing['candidate_probability_gradients']['unweighted_surface']['l1'] > 0
    checks = comparison['gradient_checks']['interior_uniform_checks']
    assert len(checks) == 3
    assert all(check['finite'] and check['correct_observed_support_directions'] for check in checks)
    assert all(check['zero_gradient_outside_observed_band'] for check in checks)


def test_interior_gradients_continuous_but_endpoint_saturation_is_preserved(comparison):
    checks = comparison['gradient_checks']
    assert checks['gradient_continuity_at_half'] and checks['largest_gradient_jump_around_half'] == 0
    assert all(row['finite'] and row['gradient_l1'] == 0 for row in checks['saturated_endpoints'])
    assert all(row['historical_clamp_saturated'] for row in checks['saturated_endpoints'])


def test_ignored_regions_invariant_against_reference_with_same_validity(comparison):
    checks = comparison['ignored_region_checks']
    assert {check['name'] for check in checks} == set(PRIOR_GATE['ignored_case_names'])
    assert all(check['target_and_validity_unchanged'] and check['reference_uses_identical_validity_and_target'] for check in checks)
    assert all(all(check['exact_invariance'].values()) for check in checks)
    assert all(all(value == 0 for value in check['differences'].values()) for check in checks)


def test_declared_gate_is_only_a_controlled_loss_screen(comparison):
    assert comparison['controlled_gate_passed']
    assert all(comparison['gate_checks'].values())
    assert comparison['scope'] == 'fixed_cpu_synthetic_candidate_screen_only'
    assert comparison['prior_gate']['no_model_or_server_gate_claim']
    assert any('not evidence' in value for value in comparison['limitations'])
    # The predeclared near checks remain visible. Their success cannot hide the
    # independently observed harmful far ordering or grant training clearance.
    assert not comparison['training_clearance']
    assert comparison['harmful_far_combined_ordering']
    assert comparison['training_blockers']
    assert comparison['no_training_on_harmful_controlled_behavior']


def test_far_error_leaving_the_band_becomes_cheaper_and_is_not_hidden(comparison, rows):
    far = comparison['far_displacement_limitations']
    assert far['dx_pixels'] == [40, 44]
    assert far['geometric_error'][1] > far['geometric_error'][0]
    assert far['historical_boundary'][0] == far['historical_boundary'][1]
    # At shift40 the erroneous rectangle still intersects the fixed radius8
    # band; at shift44 it leaves that band. The candidate does not repair this
    # far-distance failure and the report must say so.
    assert far['surface_decreases_with_larger_far_displacement']
    assert far['combined_decreases_with_larger_far_displacement']
    assert not far['surface_exact_flat']
    assert not comparison['training_clearance']
    assert rows['hard_translation_dx40']['hard_false_positive_pixels_in_surface_band'] > 0
    assert rows['hard_translation_dx44']['hard_false_positive_pixels_in_surface_band'] == 0


def test_distant_extra_building_uses_region_fallback_not_invented_surface_support(rows):
    matched, far = rows['hard_translation_dx0'], rows['extra_building_x110']
    assert far['candidate_losses']['unweighted_surface'] == matched['candidate_losses']['unweighted_surface']
    assert far['candidate_losses']['combined_region_surface'] > matched['candidate_losses']['combined_region_surface']
    assert far['hard_false_positive_pixels_in_surface_band'] == 0
    assert far['hard_false_positive_pixels_outside_surface_band'] > 0


def test_perimeter_only_and_no_support_cases_keep_finite_region_fallback(rows):
    for name in ('perimeter_only_prediction_0.05', 'perimeter_only_prediction_0.95',
                 'ignored_target_boundary_missing_building', 'no_valid_boundary_neighborhood'):
        row = rows[name]
        assert not row['surface_support']['surface_support_present']
        assert row['candidate_losses']['unweighted_surface'] == 0
        assert row['candidate_losses']['combined_region_surface'] == pytest.approx(.5 * row['historical_region'])
        assert row['candidate_probability_gradients']['unweighted_surface']['l1'] == 0


def test_support_dilution_does_not_change_class_normalized_surface_for_same_shape(rows):
    values = [rows[f'support_dilution_canvas{size}'] for size in (64, 128, 256)]
    assert all(row['candidate_losses']['unweighted_surface'] == values[0]['candidate_losses']['unweighted_surface'] for row in values)
    assert values[0]['historical_boundary'] == pytest.approx(16 * values[-1]['historical_boundary'])


def test_same_differentiable_probes_are_reported_without_generator_claim(comparison):
    probes = comparison['parameter_probes']
    assert len(probes) == 13
    assert len([row for row in probes if row['family'] == 'smooth_translation']) == 7
    assert all(row['float32_surface_runtime'] for row in probes)
    for row in probes:
        assert 'not a generator' in row['derivative_scope']
        if row['finite_difference_rule'] == 'central':
            # Float32 accumulation has a finite precision floor; use an absolute
            # tolerance, rather than pretending small finite secants are exact.
            assert row['autograd_derivatives']['unweighted_surface'] == pytest.approx(
                row['finite_difference_derivatives']['unweighted_surface'], abs=5e-5, rel=.01)
    endpoint = [row for row in probes if row['family'] == 'confidence' and row['parameter'] == 1][0]
    assert endpoint['finite_difference_rule'] == 'backward_secant_at_clamped_endpoint'
    assert endpoint['autograd_derivatives']['unweighted_surface'] == 0


def test_candidate_masks_and_probabilities_stay_cpu_and_inputs_are_preserved():
    target = rectangle((16, 16), (4, 12, 4, 12))
    valid = torch.ones_like(target)
    probability = probability_from_mask(target)
    originals = [value.clone() for value in (probability, target, valid)]
    measure_candidate(probability, target, valid)
    assert all(torch.equal(before, after) for before, after in zip(originals, (probability, target, valid)))
    with pytest.raises(ResearchError, match='already reside on CPU'):
        measure_candidate(torch.zeros(target.shape, device='meta'), target, valid)


@pytest.mark.parametrize('failure', ['nan', 'out_of_range', 'nonbinary', 'no_valid', 'wrong_shape'])
def test_comparison_rejects_malformed_inputs(failure):
    target = rectangle((8, 8), (2, 6, 2, 6))
    valid = torch.ones_like(target)
    probability = probability_from_mask(target)
    if failure == 'nan': probability[0, 0] = float('nan')
    elif failure == 'out_of_range': probability[0, 0] = 2
    elif failure == 'nonbinary': target[0, 0] = .2
    elif failure == 'no_valid': valid[:] = 0
    else: valid = valid[:-1]
    with pytest.raises(ResearchError):
        measure_candidate(probability, target, valid)


def test_fresh_json_and_partial_evidence_are_preserved(tmp_path, monkeypatch, comparison):
    import compare_surface_controlled_cases
    monkeypatch.setattr(compare_surface_controlled_cases, 'build_comparison', lambda: comparison)
    output = tmp_path / 'comparison.json'
    assert write_comparison(output) == load_json(output)
    with pytest.raises(ResearchError, match='fresh'):
        write_comparison(output)
    with pytest.raises(ResearchError, match='JSON'):
        write_comparison(tmp_path / 'wrong.txt')
    (tmp_path / 'partial.json.tmp').write_text('partial', encoding='utf-8')
    with pytest.raises(ResearchError, match='Partial'):
        write_comparison(tmp_path / 'partial.json')
    assert 'sample_id' not in output.read_text() and 'reviewer' not in output.read_text()


def test_nonfinite_comparison_never_creates_output(tmp_path, monkeypatch):
    import compare_surface_controlled_cases
    monkeypatch.setattr(compare_surface_controlled_cases, 'build_comparison', lambda: {'value': float('inf')})
    with pytest.raises(ResearchError, match='nonfinite'):
        write_comparison(tmp_path / 'invalid.json')
    assert not (tmp_path / 'invalid.json').exists()


def test_changed_candidate_policy_fails_before_output(tmp_path, monkeypatch):
    import compare_surface_controlled_cases
    monkeypatch.setattr(compare_surface_controlled_cases, 'RADIUS', 9)
    with pytest.raises(ResearchError, match='fixed prior gate'):
        write_comparison(tmp_path / 'tuned.json')
    assert not (tmp_path / 'tuned.json').exists()
