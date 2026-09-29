"""Fixed CPU comparison of the saturated valid-surface revision.

The existing 52 synthetic cases are reused without adjustment. An explicit prior
screening gate applies only to this controlled loss comparison, never to model
improvement, generalization, novelty or server allocation. No optimizer updates,
GPU work, research imagery or user annotations are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from contour_objective_cases import (
    PROTOCOL, _gradient_description, _require_finite, _support, build_report as baseline_build_report,
    fixed_cases, probability_from_mask, rectangle,
    smooth_rectangle, validate_inputs,
)
from satground.common import ResearchError, object_hash, sha256
from satground.losses import structural_losses
from saturated_surface_objective import (
    COEFFICIENT, MASS_FLOOR, POLICY, RADIUS, prepare_surface_labels,
    valid_surface_loss,
)


PRIOR_GATE = {
    'version': 'valid-surface-saturated-screen-v2',
    'scope': 'controlled_synthetic_loss_behavior_only',
    'method': 'valid_surface_saturated_v2', 'radius': 8, 'mass_floor': 32,
    'surface_coefficient': .5, 'region_coefficient': .5,
    'near_shift_pixels': [0, 1, 2, 4, 8],
    'complete_translation_pixels': [0, 1, 2, 4, 8, 16, 24, 32, 40, 44],
    'complete_translation_non_decrease': ['unweighted_surface', 'combined_region_surface'],
    'require_strict_increase': ['unweighted_surface', 'combined_region_surface'],
    'incorrect_combined_cases': [
        'hard_translation_dx4', 'uniform_probability_0', 'uniform_probability_0.05',
        'uniform_probability_0.5', 'uniform_probability_0.95', 'uniform_probability_1',
        'extra_building_x110', 'extra_thin_structure', 'inverted_reference',
    ],
    'incorrect_comparison_reference': 'matched rectangle, identical target and validity',
    'interior_uniform_probability_gradient_checks': [.05, .5, .95],
    'gradient_continuity_center_probability': .5,
    'gradient_continuity_delta': .0001,
    'continuous_gradient_domain': 'probabilities strictly inside the historical clamp; constant target support',
    'ignored_case_names': ['ignored_wrong_region', 'ignored_uncertain_region'],
    'ignored_required_invariance': ['historical_region', 'unweighted_surface', 'combined_region_surface'],
    'gradient_direction_rule': 'negative on observed building support, positive on observed nonbuilding support, zero outside support',
    'full_hard_translation_non_decrease_required': True,
    'report_saturated_distance_flatness_and_reversals': True,
    'no_model_or_server_gate_claim': True,
}
FROZEN_CASE_IDENTITY = '724d8fb9e94ef38674c1baa22a328f78e344a8dc477ab7adc36a6f2876edac77'


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def measure_candidate(probability, target, valid):
    """Use identical historical region loss; evaluate the fixed new term separately."""
    probability, target, valid = validate_inputs(probability, target, valid)
    probability = probability.detach().clone().requires_grad_(True)
    prepared = prepare_surface_labels(target.numpy(), valid.numpy())
    region, historical_boundary = structural_losses(probability[None, None], target[None, None], valid[None, None])
    surface = valid_surface_loss(probability, prepared)
    combined = .5 * region + COEFFICIENT * surface
    components = dict(unweighted_surface=surface, weighted_surface=COEFFICIENT * surface,
                      combined_region_surface=combined)
    gradients = {name: torch.autograd.grad(loss, probability, retain_graph=True)[0]
                 for name, loss in components.items()}
    region_gradient = torch.autograd.grad(region, probability)[0]
    interior, target_boundary = _support(target, valid)
    positive_boundary = (target_boundary > 0) & (interior > 0)
    surface_gradient = gradients['unweighted_surface']
    denominator = torch.linalg.vector_norm(region_gradient) * torch.linalg.vector_norm(surface_gradient)
    cosine = float((region_gradient * surface_gradient).sum() / denominator) if denominator.item() > 0 else None
    hard_prediction = probability.detach().numpy() >= .5
    truth = target.numpy().astype(bool)
    weights = prepared['weights']
    false_positive = hard_prediction & ~truth
    false_negative = ~hard_prediction & truth
    result = dict(
        historical_region=float(region.detach()), historical_boundary=float(historical_boundary.detach()),
        candidate_losses={name: float(loss.detach()) for name, loss in components.items()},
        candidate_probability_gradients={name: _gradient_description(gradient, positive_boundary)
                                        for name, gradient in gradients.items()},
        region_surface_probability_gradient_cosine=cosine,
        surface_support=dict(prepared['counts']), surface_provenance=dict(prepared['provenance']),
        hard_false_positive_pixels_in_surface_support=int((false_positive & (weights[0] > 0)).sum()),
        hard_false_positive_pixels_outside_surface_support=int((false_positive & (weights[0] == 0) & valid.numpy().astype(bool)).sum()),
        hard_false_negative_pixels_in_surface_support=int((false_negative & (weights[1] > 0)).sum()),
        hard_false_negative_pixels_outside_surface_support=int((false_negative & (weights[1] == 0) & valid.numpy().astype(bool)).sum()),
    )
    _require_finite(result)
    return result


def candidate_parameter_probe(family, value):
    """Compare analytic autograd and numerical secants on the same fixed probes."""
    require(family in ('smooth_translation', 'confidence'), 'Unknown parameter probe family.')
    target, valid = rectangle(), torch.ones((128, 128), dtype=torch.float64)
    prepared = prepare_surface_labels(target.numpy(), valid.numpy())
    parameter = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    build = smooth_rectangle if family == 'smooth_translation' else lambda p: probability_from_mask(target, p)
    probability = build(parameter)
    validate_inputs(probability, target, valid)
    region, boundary = structural_losses(probability[None, None], target[None, None], valid[None, None])
    surface = valid_surface_loss(probability, prepared)
    components = dict(historical_region=region, historical_boundary=boundary,
                      unweighted_surface=surface, combined_region_surface=.5 * region + COEFFICIENT * surface)
    derivatives = {name: float(torch.autograd.grad(loss, parameter, retain_graph=True)[0]) for name, loss in components.items()}
    delta = PROTOCOL['finite_difference_delta']
    lower, upper, difference_rule = value - delta, value + delta, 'central'
    if family == 'confidence' and upper > 1:
        lower, upper, difference_rule = value - delta, value, 'backward_secant_at_clamped_endpoint'
    scores = []
    with torch.no_grad():
        for position in (lower, upper):
            p = build(torch.tensor(position, dtype=torch.float64))
            validate_inputs(p, target, valid)
            rr, bb = structural_losses(p[None, None], target[None, None], valid[None, None])
            ss = valid_surface_loss(p, prepared)
            scores.append(dict(historical_region=float(rr), historical_boundary=float(bb), unweighted_surface=float(ss),
                               combined_region_surface=float(.5 * rr + COEFFICIENT * ss)))
    result = dict(family=family, parameter=value, values={name: float(loss.detach()) for name, loss in components.items()},
                  autograd_derivatives=derivatives,
                  finite_difference_derivatives={name: (scores[1][name] - scores[0][name]) / (upper - lower) for name in components},
                  finite_difference_rule=difference_rule,
                  float32_surface_runtime=True,
                  derivative_scope='synthetic probability-map parameter; not a generator or adapter gradient')
    _require_finite(result)
    return result


def candidate_gradient_checks():
    """Fixed uniform-map checks complement, without replacing, the 52 cases."""
    target = rectangle()
    valid = torch.ones_like(target)
    prepared = prepare_surface_labels(target.numpy(), valid.numpy())
    background_support = torch.from_numpy((prepared['weights'][0] > 0).copy())
    building_support = torch.from_numpy((prepared['weights'][1] > 0).copy())
    any_support = background_support | building_support
    gradients = []
    checks = []
    for value in PRIOR_GATE['interior_uniform_probability_gradient_checks']:
        p = torch.full_like(target, value, requires_grad=True)
        loss = valid_surface_loss(p, prepared)
        gradient = torch.autograd.grad(loss, p)[0]
        finite = bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(loss))
        directions = bool((gradient[building_support] < 0).all() and
                          (gradient[background_support] > 0).all() and
                          (gradient[~any_support] == 0).all())
        checks.append(dict(probability=value, surface_loss=float(loss.detach()), gradient_l1=float(gradient.abs().sum()),
                           finite=finite, correct_observed_support_directions=directions,
                           zero_gradient_outside_observed_seeded_components=bool((gradient[~any_support] == 0).all())))
    center, delta = PRIOR_GATE['gradient_continuity_center_probability'], PRIOR_GATE['gradient_continuity_delta']
    for value in (center - delta, center, center + delta):
        p = torch.full_like(target, value, requires_grad=True)
        gradient = torch.autograd.grad(valid_surface_loss(p, prepared), p)[0]
        gradients.append(gradient)
    largest_jump = max(float((a - b).abs().max()) for a, b in zip(gradients, gradients[1:]))
    saturated = []
    for value in (0., 1.):
        p = torch.full_like(target, value, requires_grad=True)
        loss = valid_surface_loss(p, prepared)
        gradient = torch.autograd.grad(loss, p)[0]
        saturated.append(dict(probability=value, surface_loss=float(loss.detach()), gradient_l1=float(gradient.abs().sum()),
                              historical_clamp_saturated=True, finite=bool(torch.isfinite(gradient).all())))
    result = dict(interior_uniform_checks=checks, largest_gradient_jump_around_half=largest_jump,
                  gradient_continuity_at_half=largest_jump == 0.,
                  saturated_endpoints=saturated,
                  endpoint_caveat='Retained historical clamp gives zero gradients at exact 0/1; endpoint continuity is not claimed.')
    _require_finite(result)
    return result


def _ignored_invariance(cases):
    checks = []
    for case in cases:
        if case['name'] not in PRIOR_GATE['ignored_case_names']:
            continue
        reference = measure_candidate(probability_from_mask(case['target']), case['target'], case['valid'])
        changed = measure_candidate(case['probability'], case['target'], case['valid'])
        values = dict(historical_region=(reference['historical_region'], changed['historical_region']),
                      unweighted_surface=(reference['candidate_losses']['unweighted_surface'], changed['candidate_losses']['unweighted_surface']),
                      combined_region_surface=(reference['candidate_losses']['combined_region_surface'], changed['candidate_losses']['combined_region_surface']))
        checks.append(dict(name=case['name'], exact_invariance={key: a == b for key, (a, b) in values.items()},
                           differences={key: b - a for key, (a, b) in values.items()},
                           target_and_validity_unchanged=True,
                           reference_uses_identical_validity_and_target=True))
    require(len(checks) == len(PRIOR_GATE['ignored_case_names']), 'Ignored-region checks are incomplete.')
    return checks


def _sources():
    root = Path(__file__).parents[1]
    return {str(path.relative_to(root)): sha256(path) for path in (
        root / 'src/satground/losses.py', root / 'scripts/contour_objective_cases.py',
        root / 'scripts/valid_surface_objective.py', root / 'scripts/saturated_surface_objective.py',
        root / 'docs/VALID_SURFACE_SATURATION_REVISION.md', Path(__file__),
    )}


def _build_comparison():
    require(POLICY == PRIOR_GATE['method'] and RADIUS == 8 and MASS_FLOOR == 32 and COEFFICIENT == .5,
            'Candidate policy differs from the fixed prior gate; do not silently tune it.')
    sources = _sources()
    baseline = baseline_build_report()
    cases = fixed_cases()
    require(len(cases) == baseline['synthetic_case_count'] == 52, 'Frozen case set differs.')
    require([case['name'] for case in cases] == [row['name'] for row in baseline['cases']], 'Frozen case order differs.')
    # Identity is frozen from the unchanged baseline case generator before any
    # candidate outcome is inspected. No easier cases are substituted.
    case_identity = baseline['frozen_case_identity_sha256']
    require(case_identity == FROZEN_CASE_IDENTITY, 'The prior 52-case identity changed; do not substitute a different case set.')
    gate_identity = object_hash(PRIOR_GATE)
    compared = []
    for case, historical in zip(cases, baseline['cases']):
        candidate = measure_candidate(case['probability'], case['target'], case['valid'])
        require(candidate['historical_region'] == historical['losses']['region'] and
                candidate['historical_boundary'] == historical['losses']['boundary'], 'Historical losses changed in the candidate comparison.')
        compared.append(dict(name=case['name'], family=case['family'], parameters=case['parameters'],
                             synthetic_input_hashes={key: historical[key] for key in ('probability_sha256', 'target_sha256', 'validity_sha256')},
                             historical_losses=historical['losses'], historical_probability_gradients=historical['gradients'],
                             geometry=historical['geometry'], **candidate))
    rows = {row['name']: row for row in compared}
    matched = rows['hard_translation_dx0']
    # Inversion is an explicitly fixed supplemental candidate check. It is not
    # inserted into or substituted for one of the historical 52 cases.
    reference_target = cases[0]['target']
    inverse = measure_candidate(1 - probability_from_mask(reference_target), reference_target, cases[0]['valid'])
    gradient_checks = candidate_gradient_checks()
    ignored_checks = _ignored_invariance(cases)
    shifts = [rows[f'hard_translation_dx{dx}'] for dx in PRIOR_GATE['near_shift_pixels']]
    monotonicity = {name: all(a['candidate_losses'][name] < b['candidate_losses'][name] for a, b in zip(shifts, shifts[1:]))
                    for name in PRIOR_GATE['require_strict_increase']}
    complete_shifts = [rows[f'hard_translation_dx{dx}'] for dx in PRIOR_GATE['complete_translation_pixels']]
    complete_monotonicity = {name: all(a['candidate_losses'][name] <= b['candidate_losses'][name]
                                      for a, b in zip(complete_shifts, complete_shifts[1:]))
                            for name in PRIOR_GATE['complete_translation_non_decrease']}
    translation_pairs = [dict(dx_pixels=[a['parameters']['dx_pixels'], b['parameters']['dx_pixels']],
                              candidate_surface_change=b['candidate_losses']['unweighted_surface'] - a['candidate_losses']['unweighted_surface'],
                              candidate_combined_change=b['candidate_losses']['combined_region_surface'] - a['candidate_losses']['combined_region_surface'],
                              geometric_error_change=b['geometry']['normalized_auxiliary_inner_edge_distance'] - a['geometry']['normalized_auxiliary_inner_edge_distance'])
                         for a, b in zip(complete_shifts, complete_shifts[1:])]
    incorrect = {}
    for name in PRIOR_GATE['incorrect_combined_cases']:
        row = inverse if name == 'inverted_reference' else rows[name]
        incorrect[name] = row['candidate_losses']['combined_region_surface'] > matched['candidate_losses']['combined_region_surface']
    gates = dict(near_shift_strict_increase=all(monotonicity.values()), incorrect_combined_above_matched=all(incorrect.values()),
                 complete_hard_translation_non_decrease=all(complete_monotonicity.values()),
                 finite_correct_continuous_interior_gradients=all(row['finite'] and row['correct_observed_support_directions']
                                                                for row in gradient_checks['interior_uniform_checks'])
                                                            and gradient_checks['gradient_continuity_at_half'],
                 ignored_regions_invariant=all(all(row['exact_invariance'].values()) for row in ignored_checks))
    far_40, far_44 = rows['hard_translation_dx40'], rows['hard_translation_dx44']
    far = dict(dx_pixels=[40, 44], candidate_surface=[r['candidate_losses']['unweighted_surface'] for r in (far_40, far_44)],
               candidate_combined=[r['candidate_losses']['combined_region_surface'] for r in (far_40, far_44)],
               historical_boundary=[r['historical_boundary'] for r in (far_40, far_44)],
               geometric_error=[r['geometry']['normalized_auxiliary_inner_edge_distance'] for r in (far_40, far_44)],
               surface_exact_flat=far_40['candidate_losses']['unweighted_surface'] == far_44['candidate_losses']['unweighted_surface'],
               surface_decreases_with_larger_far_displacement=far_44['candidate_losses']['unweighted_surface'] < far_40['candidate_losses']['unweighted_surface'],
               combined_decreases_with_larger_far_displacement=far_44['candidate_losses']['combined_region_surface'] < far_40['candidate_losses']['combined_region_surface'],
               caveat='Weights saturate at one beyond distance8 within observed seeded components. Region BCE and saturated weights can plateau after nonoverlap.')
    harmful_far_ordering = (far['geometric_error'][1] > far['geometric_error'][0]
                           and far['combined_decreases_with_larger_far_displacement'])
    blockers = []
    if harmful_far_ordering:
        blockers.append('Combined structural loss decreases while fixed far-displacement geometric error grows.')
    if not all(gates.values()):
        blockers.append('At least one unchanged prior controlled screening gate failed.')
    probes = [candidate_parameter_probe('smooth_translation', value) for value in PROTOCOL['smooth_translation_pixels']]
    probes += [candidate_parameter_probe('confidence', value) for value in PROTOCOL['confidence_contrasts']]
    require(_sources() == sources, 'A compared source changed during the CPU diagnostic; preserve outputs and rerun to a fresh path.')
    result = dict(scope='fixed_cpu_synthetic_saturated_candidate_screen_only', fixed_case_identity_sha256=case_identity,
                  fixed_synthetic_case_count=52, case_order_unchanged=True, historical_losses_reproduce_exactly=True,
                  prior_gate=PRIOR_GATE, prior_gate_identity_sha256=gate_identity, gate_checks=gates,
                  controlled_gate_passed=all(gates.values()), near_shift_monotonicity=monotonicity,
                  complete_hard_translation_monotonicity=complete_monotonicity,
                  complete_hard_translation_pairs=translation_pairs,
                  controlled_gate_passed_scope='Fixed near shifts, complete translation family, incorrect references, interior gradients and ignored regions.',
                  harmful_far_combined_ordering=harmful_far_ordering,
                  training_clearance=all(gates.values()) and not harmful_far_ordering,
                  training_clearance_scope='Controlled-case prerequisite only; never an authorization or evidence of model improvement.',
                  training_blockers=blockers,
                  no_training_on_harmful_controlled_behavior=True,
                  incorrect_case_checks=incorrect, inversion_supplemental_check=inverse,
                  gradient_checks=gradient_checks, ignored_region_checks=ignored_checks,
                  parameter_probes=probes,
                  far_displacement_limitations=far, cases=compared, source_hashes=sources,
                  versions={key: baseline[key] for key in ('torch_version', 'numpy_version', 'scipy_version')},
                  gpu_used=False, optimizer_updates=0, real_data_read=False,
                  generator_improvement_established=False, server_gate_eligible=False,
                  limitations=[
                      'Passing controlled loss gates is not evidence that a generator or adapter improves.',
                      'All fixed gates, including the complete hard-translation family, must pass; harmful ordering or nonfinite behavior blocks training.',
                      'The same 52 prior cases are reused; inversion and uniform-gradient checks are separate declared checks.',
                      'Probability-map gradients omit frozen-segmenter and generator Jacobians.',
                      'Distance weights saturate at8; arbitrarily far displacement may plateau after nonoverlap, without an unlimited distance ranking claim.',
                      'Surface weights cover observed seeded components; unchanged region BCE controls unobserved components and has no displacement ordering after nonoverlap.',
                      'The historical probability clamp retains endpoint saturation; continuity is assessed only in its interior.',
                      'Surface runtime is float32; numerical finite secants can differ from autograd because of roundoff.',
                      'No loss tuning, candidate selection, training, real-data eligibility or held-out evaluation occurs here.',
                  ])
    _require_finite(result)
    return result


def build_comparison():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _build_comparison()
    finally:
        torch.set_num_threads(previous)


def build_report():
    """Deterministic CPU-only aggregate API for zero-update preflight checks."""
    return build_comparison()


def write_comparison(output):
    output = Path(output)
    require(output.suffix == '.json', 'Comparison output must be JSON.')
    require(not output.exists(), 'Use a fresh comparison path; preserve existing evidence.')
    require(not output.with_suffix(output.suffix + '.tmp').exists(), 'Partial output exists; preserve it and use a fresh path.')
    result = build_comparison()
    _require_finite(result)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = write_comparison(args.output)
    print(f"Compared {result['fixed_synthetic_case_count']} unchanged CPU cases; prior gate passed={result['controlled_gate_passed']}; controlled training clearance={result['training_clearance']}; no training.")
