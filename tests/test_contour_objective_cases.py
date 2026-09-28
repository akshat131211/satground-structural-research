"""Synthetic loss-behavior regressions, including known current-loss limitations.

A geometry-ordering or collapse limitation is measured evidence, not treated as
an implementation bug that changes the frozen loss to make this test pass.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from contour_objective_cases import (
    PROTOCOL, build_report, fixed_cases, measure_case, parameter_derivative,
    probability_from_mask, rectangle, validate_inputs, write_report,
)
from satground.common import ResearchError, load_json


@pytest.fixture(scope='module')
def report():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield build_report()
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture(scope='module')
def rows(report):
    return {row['name']: row for row in report['cases']}


def test_fixed_protocol_contains_only_cpu_synthetic_inputs(report):
    assert report['scope'] == 'fixed_cpu_synthetic_objective_diagnosis_only'
    assert report['protocol']['region_weight'] == report['protocol']['boundary_weight'] == .5
    assert not report['real_data_read'] and not report['gpu_used']
    assert report['optimizer_updates'] == 0 and not report['generator_improvement_established']
    assert len(report['frozen_case_identity_sha256']) == 64
    assert report['synthetic_case_count'] == len(report['cases'])
    assert report['differentiable_probe_count'] == 13
    assert all(case['probability'].device.type == 'cpu' for case in fixed_cases())


def test_small_shifts_worsen_known_geometry_and_current_loss(rows):
    keys = [f'hard_translation_dx{shift}' for shift in (0, 1, 2, 4, 8, 16)]
    assert all(rows[a]['geometry']['hard_building_iou'] > rows[b]['geometry']['hard_building_iou']
               for a, b in zip(keys, keys[1:]))
    assert all(rows[a]['geometry']['normalized_auxiliary_inner_edge_distance'] <
               rows[b]['geometry']['normalized_auxiliary_inner_edge_distance'] for a, b in zip(keys, keys[1:]))
    for component in ('region', 'boundary', 'A3_structural'):
        assert all(rows[a]['losses'][component] < rows[b]['losses'][component] for a, b in zip(keys, keys[1:]))


def test_far_displacement_plateau_is_recorded_as_current_loss_limitation(rows):
    close, farther = rows['hard_translation_dx40'], rows['hard_translation_dx44']
    assert close['geometry']['normalized_auxiliary_inner_edge_distance'] < farther['geometry']['normalized_auxiliary_inner_edge_distance']
    for component in ('region', 'boundary', 'A3_structural'):
        assert close['losses'][component] == pytest.approx(farther['losses'][component], abs=1e-14)


def test_missing_building_can_have_less_boundary_penalty_than_misplaced_building(rows):
    missing = rows['uniform_probability_0.05']
    misplaced = rows['hard_translation_dx40']
    assert missing['geometry']['hard_building_iou'] == misplaced['geometry']['hard_building_iou'] == 0
    assert missing['losses']['boundary'] < misplaced['losses']['boundary']
    assert missing['losses']['A3_structural'] < misplaced['losses']['A3_structural']
    # A matched outline is still preferred by the combined loss. The empty map
    # lacks the extra false-positive region of the displaced map; this comparison
    # is not evidence that optimizing a generator will collapse to an empty mask.
    assert rows['hard_translation_dx0']['losses']['A3_structural'] < missing['losses']['A3_structural']
    assert missing['losses']['boundary'] > 0
    # Max/min pooling picks matching locations on a constant field; the current
    # boundary derivative cannot create an outline from this exact uniform map.
    assert missing['gradients']['boundary']['l1'] == 0
    assert missing['gradients']['region']['l1'] > 0


def test_probability_clamp_saturates_gradients_on_wrong_confident_geometry(rows):
    saturated = rows['saturated_wrong_translation']
    assert saturated['losses']['region'] > rows['hard_translation_dx40']['losses']['region']
    assert saturated['geometry']['hard_building_iou'] == 0
    for component in ('region', 'boundary', 'A3_structural'):
        assert saturated['gradients'][component]['l1'] == 0
    assert rows['hard_translation_dx40']['gradients']['region']['l1'] > 0


def test_confidence_can_improve_losses_without_moving_outline(rows):
    lower, higher = rows['confidence_contrast_0.2'], rows['confidence_contrast_0.98']
    assert lower['geometry'] == higher['geometry']
    assert lower['geometry']['hard_building_iou'] == 1
    assert lower['losses']['region'] > higher['losses']['region']
    assert lower['losses']['boundary'] > higher['losses']['boundary']


def test_extra_building_distance_does_not_change_current_loss(rows):
    near, far = rows['extra_building_x84'], rows['extra_building_x110']
    assert near['geometry']['normalized_auxiliary_inner_edge_distance'] < far['geometry']['normalized_auxiliary_inner_edge_distance']
    for component in ('region', 'boundary', 'A3_structural'):
        assert near['losses'][component] == pytest.approx(far['losses'][component], abs=1e-14)
    assert far['losses']['boundary'] > rows['hard_translation_dx0']['losses']['boundary']


def test_erosion_dilation_and_thin_extra_contours_are_measured(rows):
    matched = rows['hard_translation_dx0']
    for name in ('eroded_radius_1', 'dilated_radius_1', 'blurred_edge_sigma_2', 'extra_thin_structure'):
        assert rows[name]['losses']['boundary'] > matched['losses']['boundary']
        assert rows[name]['gradients']['A3_structural']['l1'] > 0
    assert rows['eroded_radius_16']['geometry']['hard_predicted_building_pixels'] == 0
    assert rows['extra_building_side16']['losses']['boundary'] > rows['extra_building_side4']['losses']['boundary']


def test_perimeter_only_target_has_no_training_outline(rows):
    missing, present = rows['perimeter_only_prediction_0.05'], rows['perimeter_only_prediction_0.95']
    assert missing['positive_target_contour_pixels'] == present['positive_target_contour_pixels'] == 0
    assert present['geometry']['auxiliary_target_image_perimeter_edge_pixels'] > 0
    assert missing['losses']['boundary'] == present['losses']['boundary'] == 0
    assert missing['gradients']['boundary']['l1'] == present['gradients']['boundary']['l1'] == 0
    assert missing['losses']['region'] > present['losses']['region']


def test_unchanged_outline_penalty_dilutes_with_more_scored_background(rows):
    small, medium, large = [rows[f'support_dilution_canvas{size}'] for size in (64, 128, 256)]
    assert small['positive_target_contour_pixels'] == medium['positive_target_contour_pixels'] == large['positive_target_contour_pixels']
    assert small['losses']['boundary'] == pytest.approx(4 * medium['losses']['boundary'])
    assert medium['losses']['boundary'] == pytest.approx(4 * large['losses']['boundary'])
    assert large['positive_target_contour_fraction'] < .002


def test_ignored_pixels_do_not_restore_structure_supervision(rows):
    matched = rows['hard_translation_dx0']
    for name in ('ignored_wrong_region', 'ignored_uncertain_region'):
        # The valid denominator changes, so the small matched contour penalty
        # changes by that denominator, while wrong/uncertain pixels stay ignored.
        assert rows[name]['losses']['region'] == pytest.approx(matched['losses']['region'])
        assert rows[name]['losses']['boundary'] * rows[name]['boundary_valid_pixels'] == pytest.approx(
            matched['losses']['boundary'] * matched['boundary_valid_pixels'])
    assert rows['scored_uncertain_region']['losses']['region'] > matched['losses']['region']
    hidden = rows['ignored_target_boundary_missing_building']
    assert hidden['positive_target_contour_pixels'] == 0
    assert hidden['losses']['boundary'] == 0
    none = rows['no_valid_boundary_neighborhood']
    assert none['region_valid_pixels'] == 1 and none['boundary_valid_pixels'] == 0
    assert none['no_boundary_support'] and none['losses']['boundary'] == 0


def test_weighted_components_remain_separate_and_exact(rows):
    for row in rows.values():
        losses = row['losses']
        assert losses['weighted_region'] == pytest.approx(.5 * losses['region'])
        assert losses['weighted_boundary'] == pytest.approx(.5 * losses['boundary'])
        assert losses['A2_structural'] == losses['weighted_region']
        assert losses['A3_structural'] == pytest.approx(losses['weighted_region'] + losses['weighted_boundary'])
        assert row['gradients']['weighted_boundary']['l1'] == pytest.approx(.5 * row['gradients']['boundary']['l1'])


def test_smooth_translation_derivatives_match_local_finite_differences(report):
    probes = [probe for probe in report['parameter_derivatives'] if probe['family'] == 'smooth_translation']
    assert len(probes) == 7
    for probe in probes:
        assert probe['finite_difference_rule'] == 'central'
        for component in ('region', 'boundary', 'A3_structural'):
            assert probe['autograd_derivative'][component] == pytest.approx(
                probe['finite_difference_derivative'][component], rel=.005, abs=1e-7)


def test_confidence_endpoint_secant_is_not_claimed_to_be_local_derivative(report):
    probes = [probe for probe in report['parameter_derivatives'] if probe['family'] == 'confidence']
    for probe in probes[:-1]:
        for component in ('region', 'boundary', 'A3_structural'):
            assert probe['autograd_derivative'][component] < 0
            assert probe['autograd_derivative'][component] == pytest.approx(
                probe['finite_difference_derivative'][component], rel=.005, abs=1e-7)
    endpoint = probes[-1]
    assert endpoint['parameter'] == 1
    assert endpoint['finite_difference_rule'] == 'backward_secant_at_clamped_endpoint'
    assert endpoint['autograd_derivative']['region'] == 0
    assert endpoint['finite_difference_derivative']['region'] < 0


@pytest.mark.parametrize('which,value,message', [
    ('probability', float('nan'), 'nonfinite'), ('probability', float('inf'), 'nonfinite'),
    ('probability', -1., 'lie'), ('probability', 1.1, 'lie'),
    ('target', .25, 'binary'), ('valid', .25, 'binary'),
])
def test_nonfinite_out_of_range_and_nonbinary_inputs_fail_closed(which, value, message):
    target = rectangle()
    inputs = dict(probability=probability_from_mask(target), target=target, valid=torch.ones_like(target))
    inputs[which] = inputs[which].clone()
    inputs[which][0, 0] = value
    with pytest.raises(ResearchError, match=message):
        validate_inputs(**inputs)


def test_empty_validity_wrong_shapes_and_tiny_images_fail_closed():
    target = rectangle()
    with pytest.raises(ResearchError, match='No scored'):
        validate_inputs(probability_from_mask(target), target, torch.zeros_like(target))
    with pytest.raises(ResearchError, match='shapes differ'):
        validate_inputs(torch.zeros((64, 64)), target, torch.ones_like(target))
    with pytest.raises(ResearchError, match='2-D'):
        validate_inputs(torch.zeros((2, 2)), torch.zeros((2, 2)), torch.ones((2, 2)))
    with pytest.raises(ResearchError, match='2-D'):
        validate_inputs(torch.zeros((1, 4, 4)), torch.zeros((1, 4, 4)), torch.ones((1, 4, 4)))


def test_non_cpu_input_rejected_without_transferring_it():
    target = rectangle()
    with pytest.raises(ResearchError, match='already reside on CPU'):
        validate_inputs(torch.zeros((128, 128), device='meta'), target, torch.ones_like(target))


def test_complex_and_nonnumeric_inputs_rejected():
    target = rectangle()
    with pytest.raises(ResearchError, match='real-valued'):
        validate_inputs(torch.zeros_like(target, dtype=torch.complex128), target, torch.ones_like(target))
    with pytest.raises(ResearchError, match='numeric CPU array'):
        validate_inputs({'wrong': 1}, target, torch.ones_like(target))


def test_cpu_diagnostic_does_not_call_cuda(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError('Synthetic diagnostics must not call CUDA.')
    monkeypatch.setattr(torch.cuda, 'is_available', prohibited)
    monkeypatch.setattr(torch.cuda, 'init', prohibited)
    target = rectangle((8, 8), (2, 6, 2, 6))
    result = measure_case(probability_from_mask(target), target, torch.ones_like(target))
    assert result['geometry']['hard_building_iou'] == 1


def test_ignored_input_gradients_are_zero():
    target = rectangle()
    valid = torch.ones_like(target)
    valid[100:121, 100:121] = 0
    probability = probability_from_mask(target).requires_grad_(True)
    from satground.losses import structural_losses
    region, boundary = structural_losses(probability[None, None], target[None, None], valid[None, None])
    region_grad = torch.autograd.grad(region, probability, retain_graph=True)[0]
    boundary_grad = torch.autograd.grad(boundary, probability)[0]
    assert torch.count_nonzero(region_grad[104:117, 104:117]) == 0
    assert torch.count_nonzero(boundary_grad[104:117, 104:117]) == 0


def test_output_freshness_and_synthetic_report_identity(tmp_path, monkeypatch, report):
    import contour_objective_cases
    monkeypatch.setattr(contour_objective_cases, 'build_report', lambda: report)
    output = tmp_path / 'result.json'
    assert write_report(output) == load_json(output)
    payload = output.read_text()
    assert 'sample_id' not in payload and 'image_path' not in payload and 'reviewer' not in payload
    with pytest.raises(ResearchError, match='fresh'):
        write_report(output)
    with pytest.raises(ResearchError, match='JSON'):
        write_report(tmp_path / 'wrong.txt')
    partial = tmp_path / 'partial.json.tmp'
    partial.write_text('partial', encoding='utf-8')
    with pytest.raises(ResearchError, match='partial'):
        write_report(tmp_path / 'partial.json')


def test_nonfinite_report_fails_before_output_creation(tmp_path, monkeypatch):
    import contour_objective_cases
    monkeypatch.setattr(contour_objective_cases, 'build_report', lambda: {'bad': float('nan')})
    with pytest.raises(ResearchError, match='nonfinite'):
        write_report(tmp_path / 'nonfinite.json')
    assert not (tmp_path / 'nonfinite.json').exists()
