"""Fixed CPU-only sensitivity cases for the current structural objective.

These synthetic measurements diagnose loss behavior. They never train a model,
read research imagery or select a revised loss. Geometry changes of hard masks
are reported as finite comparisons, not differentiable translation derivatives.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
import torch

from satground.common import ResearchError, object_hash, sha256
from satground.losses import soft_boundary, structural_losses


# The complete case set is defined before measurements. No result-dependent
# cases, loss weights, thresholds or randomized scenes are added by this command.
PROTOCOL = {
    'version': 'current-contour-synthetic-v1',
    'device': 'cpu', 'dtype': 'float64', 'cpu_threads': 1, 'canvas': [128, 128],
    'main_rectangle_yx': [48, 80, 48, 80],
    'foreground_probability': .95, 'background_probability': .05,
    'region_weight': .5, 'boundary_weight': .5,
    'hard_translation_pixels': [0, 1, 2, 4, 8, 16, 24, 32, 40, 44],
    'smooth_translation_pixels': [.25, 1.25, 2.25, 4.25, 8.25, 16.25, 40.25],
    'smooth_rectangle_sigmoid_width_pixels': 1.,
    'confidence_contrasts': [.2, .5, .8, .9, .98, 1.],
    'uniform_probabilities': [0., .05, .5, .95, 1.],
    'erosion_pixels': [1, 2, 4, 8, 12, 16],
    'dilation_pixels': [1, 2, 4, 8],
    'blur_sigma_pixels': [.5, 1., 2., 4.],
    'extra_rectangle_x': [84, 110],
    'support_canvas_sizes': [64, 128, 256],
    'support_rectangle_side_pixels': 8,
    'finite_difference_delta': .001,
    'boundary_support_rule': 'unchanged legacy 3x3 valid-interior erosion',
    'hard_mask_threshold': .5,
    'gradient_zero_tolerance': 1e-12,
    'case_families': [
        'hard_translation', 'uniform_prediction', 'confidence_only',
        'erosion', 'dilation', 'blurred_edge', 'extra_building_distance',
        'extra_building_size', 'extra_thin_structure', 'perimeter_only',
        'support_dilution', 'ignored_wrong_region', 'ignored_uncertain_region',
        'scored_uncertain_region', 'ignored_target_boundary',
        'no_valid_boundary_neighborhood', 'saturated_wrong_translation',
        'smooth_translation_derivative', 'confidence_derivative',
    ],
}


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def validate_inputs(probability, target, valid):
    """Reject malformed inputs rather than silently rescaling synthetic masks."""
    arrays = []
    for name, value in [('probability', probability), ('target', target), ('valid', valid)]:
        if torch.is_tensor(value):
            require(value.device.type == 'cpu', f'{name} must already reside on CPU; GPU transfers are disallowed.')
        try:
            value = torch.as_tensor(value, device='cpu')
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ResearchError(f'{name} must be a numeric CPU array.') from exc
        require(not value.is_complex(), f'{name} must be real-valued.')
        require(value.ndim == 2 and min(value.shape) >= 3, f'{name} must be a 2-D image of at least 3x3.')
        require(torch.isfinite(value).all().item(), f'{name} contains nonfinite values.')
        require(((value >= 0) & (value <= 1)).all().item(), f'{name} must lie in [0, 1].')
        if name != 'probability':
            require(((value == 0) | (value == 1)).all().item(), f'{name} must be binary.')
        arrays.append(value.to(dtype=torch.float64))
    require(arrays[0].shape == arrays[1].shape == arrays[2].shape, 'Probability, target and validity shapes differ.')
    require(arrays[2].sum().item() > 0, 'No scored region pixels.')
    return arrays


def rectangle(shape=(128, 128), bounds=(48, 80, 48, 80)):
    y0, y1, x0, x1 = bounds
    require(0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1], 'Rectangle leaves its canvas.')
    result = torch.zeros(shape, dtype=torch.float64, device='cpu')
    result[y0:y1, x0:x1] = 1
    return result


def probability_from_mask(mask, contrast=.9):
    return .5 + contrast * (mask - .5)


def _as_batch(value):
    return value[None, None]


def _support(target, valid):
    interior = -torch.nn.functional.max_pool2d(-_as_batch(valid), 3, stride=1, padding=1)
    target_edge = soft_boundary(_as_batch(target))
    return interior[0, 0], target_edge[0, 0]


def _geometric_descriptions(probability, target, valid):
    # Auxiliary geometric descriptions never feed structural_losses. This
    # erosion closes masks at the image perimeter, unlike the training gradient.
    predicted = (probability.detach().numpy() >= .5) & valid.numpy().astype(bool)
    truth = target.numpy().astype(bool) & valid.numpy().astype(bool)
    union = np.count_nonzero(predicted | truth)
    pe = predicted ^ binary_erosion(predicted, structure=np.ones((3, 3)), border_value=0)
    te = truth ^ binary_erosion(truth, structure=np.ones((3, 3)), border_value=0)
    # Require valid neighbors; crop-closing edges are counted separately.
    interior = binary_erosion(valid.numpy().astype(bool), structure=np.ones((3, 3)), border_value=1)
    pe, te = pe & interior, te & interior
    perimeter = np.zeros_like(te)
    perimeter[[0, -1], :] = True
    perimeter[:, [0, -1]] = True
    if not pe.any() and not te.any():
        distance, case = 0., 'both_empty'
    elif not pe.any() or not te.any():
        distance, case = 1., 'one_empty'
    else:
        distance = float(.5 * (distance_transform_edt(~pe)[te].mean() + distance_transform_edt(~te)[pe].mean())
                         / np.hypot(*te.shape))
        case = 'neither_empty'
    return dict(hard_building_iou=float(np.count_nonzero(predicted & truth) / union) if union else 1.,
                normalized_auxiliary_inner_edge_distance=distance,
                auxiliary_empty_edge_case=case,
                auxiliary_target_inner_edge_pixels=int(te.sum()),
                auxiliary_target_image_perimeter_edge_pixels=int((te & perimeter).sum()),
                hard_predicted_building_pixels=int(predicted.sum()), hard_target_building_pixels=int(truth.sum()))


def _gradient_description(gradient, positive_contour):
    abs_gradient = gradient.abs()
    tolerance = PROTOCOL['gradient_zero_tolerance']
    return dict(l1=float(abs_gradient.sum()), l2=float(torch.linalg.vector_norm(gradient)),
                maximum_absolute=float(abs_gradient.max()), signed_sum=float(gradient.sum()),
                positive_gradient_pixels=int((gradient > tolerance).sum()),
                negative_gradient_pixels=int((gradient < -tolerance).sum()),
                l1_at_positive_target_contour=float(abs_gradient[positive_contour].sum()),
                l1_away_from_positive_target_contour=float(abs_gradient[~positive_contour].sum()))


def measure_case(probability, target, valid):
    """Measure current losses and derivatives with respect to input probabilities."""
    probability, target, valid = validate_inputs(probability, target, valid)
    probability = probability.detach().clone().requires_grad_(True)
    region, boundary = structural_losses(_as_batch(probability), _as_batch(target), _as_batch(valid))
    components = dict(region=region, boundary=boundary,
                      weighted_region=.5 * region, weighted_boundary=.5 * boundary,
                      A2_structural=.5 * region, A3_structural=.5 * region + .5 * boundary)
    gradients = {name: torch.autograd.grad(loss, probability, retain_graph=True)[0]
                 for name, loss in components.items()}
    interior, target_edge = _support(target, valid)
    positive_contour = (target_edge > 0) & (interior > 0)
    gradient_region, gradient_boundary = gradients['region'], gradients['boundary']
    product = torch.linalg.vector_norm(gradient_region) * torch.linalg.vector_norm(gradient_boundary)
    alignment = float((gradient_region * gradient_boundary).sum() / product) if product.item() > 0 else None
    result = dict(losses={name: float(value.detach()) for name, value in components.items()},
                  gradients={name: _gradient_description(value, positive_contour) for name, value in gradients.items()},
                  region_boundary_probability_gradient_cosine=alignment,
                  region_valid_pixels=int(valid.sum()), boundary_valid_pixels=int(interior.sum()),
                  positive_target_contour_pixels=int(positive_contour.sum()),
                  positive_target_contour_fraction=float(positive_contour.sum() / interior.sum()) if interior.sum() else None,
                  no_boundary_support=not bool(interior.any()),
                  geometry=_geometric_descriptions(probability, target, valid))
    _require_finite(result)
    return result


def fixed_cases():
    """Build a deterministic set without inspecting any measured loss."""
    shape = tuple(PROTOCOL['canvas'])
    target = rectangle(shape, tuple(PROTOCOL['main_rectangle_yx']))
    valid = torch.ones_like(target)
    matched = probability_from_mask(target)
    cases = []

    def add(name, family, probability, case_target=target, case_valid=valid, **parameters):
        cases.append(dict(name=name, family=family, parameters=parameters,
                          probability=probability, target=case_target.clone(), valid=case_valid.clone()))

    for dx in PROTOCOL['hard_translation_pixels']:
        predicted = rectangle(shape, (48, 80, 48 + dx, 80 + dx))
        add(f'hard_translation_dx{dx}', 'hard_translation', probability_from_mask(predicted), dx_pixels=dx)
    for value in PROTOCOL['uniform_probabilities']:
        add(f'uniform_probability_{value:g}', 'uniform_prediction', torch.full_like(target, value), probability_level=value)
    for contrast in PROTOCOL['confidence_contrasts']:
        add(f'confidence_contrast_{contrast:g}', 'confidence_only', probability_from_mask(target, contrast), contrast=contrast,
            hard_geometry_unchanged_for_positive_contrast=True)
    for radius in PROTOCOL['erosion_pixels']:
        predicted = rectangle(shape, (48 + radius, 80 - radius, 48 + radius, 80 - radius)) if radius < 16 else torch.zeros_like(target)
        add(f'eroded_radius_{radius}', 'erosion', probability_from_mask(predicted), radius_pixels=radius)
    for radius in PROTOCOL['dilation_pixels']:
        predicted = rectangle(shape, (48 - radius, 80 + radius, 48 - radius, 80 + radius))
        add(f'dilated_radius_{radius}', 'dilation', probability_from_mask(predicted), radius_pixels=radius)
    for sigma in PROTOCOL['blur_sigma_pixels']:
        blurred = torch.as_tensor(gaussian_filter(target.numpy(), sigma=sigma, mode='nearest'), dtype=torch.float64)
        add(f'blurred_edge_sigma_{sigma:g}', 'blurred_edge', probability_from_mask(blurred), sigma_pixels=sigma)
    for x0 in PROTOCOL['extra_rectangle_x']:
        predicted = torch.maximum(target, rectangle(shape, (56, 64, x0, x0 + 8)))
        add(f'extra_building_x{x0}', 'extra_building_distance', probability_from_mask(predicted), extra_building_x=x0,
            extra_building_side_pixels=8)
    for side in (4, 8, 16):
        predicted = torch.maximum(target, rectangle(shape, (48, 48 + side, 104, 104 + side)))
        add(f'extra_building_side{side}', 'extra_building_size', probability_from_mask(predicted), extra_building_side_pixels=side)
    predicted = torch.maximum(target, rectangle(shape, (52, 76, 110, 111)))
    add('extra_thin_structure', 'extra_thin_structure', probability_from_mask(predicted), extra_structure_shape=[24, 1])
    full = torch.ones_like(target)
    for value in (.05, .95):
        add(f'perimeter_only_prediction_{value:g}', 'perimeter_only', torch.full_like(target, value), case_target=full,
            probability_level=value, no_internal_target_outline=True)
    for canvas_size in PROTOCOL['support_canvas_sizes']:
        center = canvas_size // 2
        sparse_target = rectangle((canvas_size, canvas_size), (center - 4, center + 4, center - 4, center + 4))
        sparse_valid = torch.ones_like(sparse_target)
        add(f'support_dilution_canvas{canvas_size}', 'support_dilution', torch.full_like(sparse_target, .05),
            case_target=sparse_target, case_valid=sparse_valid, canvas_size=canvas_size, target_side_pixels=8)
    ignored = valid.clone()
    ignored[100:121, 100:121] = 0
    wrong = matched.clone()
    wrong[104:117, 104:117] = .95
    add('ignored_wrong_region', 'ignored_wrong_region', wrong, case_valid=ignored)
    uncertain = matched.clone()
    uncertain[104:117, 104:117] = .5
    add('ignored_uncertain_region', 'ignored_uncertain_region', uncertain, case_valid=ignored)
    add('scored_uncertain_region', 'scored_uncertain_region', uncertain)
    hidden = valid.clone()
    hidden[46:82, 46:82] = 0
    add('ignored_target_boundary_missing_building', 'ignored_target_boundary', torch.full_like(target, .05), case_valid=hidden)
    no_neighborhood = torch.zeros_like(target)
    no_neighborhood[64, 64] = 1
    add('no_valid_boundary_neighborhood', 'no_valid_boundary_neighborhood', torch.full_like(target, .05), case_valid=no_neighborhood)
    wrong_hard = rectangle(shape, (48, 80, 88, 120))
    add('saturated_wrong_translation', 'saturated_wrong_translation', wrong_hard, dx_pixels=40)
    return cases


def smooth_rectangle(dx, shape=(128, 128)):
    """Smooth translation parameter, unlike a hard mask roll/slice."""
    dtype, device = dx.dtype, dx.device
    y = torch.arange(shape[0], dtype=dtype, device=device)[:, None] + .5
    x = torch.arange(shape[1], dtype=dtype, device=device)[None, :] + .5
    width = PROTOCOL['smooth_rectangle_sigmoid_width_pixels']
    mask = (torch.sigmoid((x - 48 - dx) / width) * torch.sigmoid((80 + dx - x) / width)
            * torch.sigmoid((y - 48) / width) * torch.sigmoid((80 - y) / width))
    return probability_from_mask(mask)


def parameter_derivative(family, value):
    require(family in ('smooth_translation', 'confidence'), 'Unknown differentiable parameter family.')
    target = rectangle()
    valid = torch.ones_like(target)
    parameter = torch.tensor(value, dtype=torch.float64, device='cpu', requires_grad=True)
    build = smooth_rectangle if family == 'smooth_translation' else lambda p: probability_from_mask(target, p)
    probability = build(parameter)
    validate_inputs(probability, target, valid)
    region, boundary = structural_losses(_as_batch(probability), _as_batch(target), _as_batch(valid))
    components = dict(region=region, boundary=boundary, weighted_region=.5 * region,
                      weighted_boundary=.5 * boundary, A2_structural=.5 * region,
                      A3_structural=.5 * region + .5 * boundary)
    gradients = {name: float(torch.autograd.grad(loss, parameter, retain_graph=True)[0])
                 for name, loss in components.items()}
    delta = PROTOCOL['finite_difference_delta']
    lower, upper = value - delta, value + delta
    difference_rule = 'central'
    if family == 'confidence' and upper > 1:
        upper, lower, difference_rule = value, value - delta, 'backward_secant_at_clamped_endpoint'
    scores = []
    with torch.no_grad():
        for position in (lower, upper):
            prob = build(torch.tensor(position, dtype=torch.float64, device='cpu'))
            validate_inputs(prob, target, valid)
            rr, bb = structural_losses(_as_batch(prob), _as_batch(target), _as_batch(valid))
            scores.append(dict(region=float(rr), boundary=float(bb), weighted_region=float(.5 * rr),
                               weighted_boundary=float(.5 * bb), A2_structural=float(.5 * rr),
                               A3_structural=float(.5 * rr + .5 * bb)))
    result = dict(family=family, parameter=value, losses={name: float(loss.detach()) for name, loss in components.items()},
                  autograd_derivative=gradients,
                  finite_difference_derivative={name: (scores[1][name] - scores[0][name]) / (upper - lower) for name in components},
                  finite_difference_rule=difference_rule,
                  derivative_scope='probability-map parameter only; not a generator-parameter derivative',
                  nondifferentiability_note='Max/min pooling and absolute loss are piecewise differentiable; endpoints cross the probability clamp.')
    _require_finite(result)
    return result


def _require_finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'A nonfinite diagnostic number was produced.')
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)


def _tensor_hash(value):
    return hashlib.sha256(value.detach().numpy().astype('<f8').tobytes()).hexdigest()


def _build_report():
    cases = fixed_cases()
    names = [case['name'] for case in cases]
    require(len(names) == len(set(names)), 'Duplicate synthetic case names.')
    definition = [dict(name=case['name'], family=case['family'], parameters=case['parameters'],
                       shape=list(case['target'].shape),
                       probability_sha256=_tensor_hash(case['probability']), target_sha256=_tensor_hash(case['target']),
                       validity_sha256=_tensor_hash(case['valid'])) for case in cases]
    # Freeze identity from all generated arrays before evaluating any losses.
    frozen_identity = object_hash(dict(protocol=PROTOCOL, cases=definition))
    measured = [dict(definition[i], **measure_case(case['probability'], case['target'], case['valid']))
                for i, case in enumerate(cases)]
    derivatives = [parameter_derivative('smooth_translation', value) for value in PROTOCOL['smooth_translation_pixels']]
    derivatives += [parameter_derivative('confidence', value) for value in PROTOCOL['confidence_contrasts']]
    return dict(scope='fixed_cpu_synthetic_objective_diagnosis_only', protocol=PROTOCOL,
                frozen_case_identity_sha256=frozen_identity, synthetic_case_count=len(cases),
                differentiable_probe_count=len(derivatives), cases=measured, parameter_derivatives=derivatives,
                source_hashes={'structural_losses': sha256(Path(__file__).parents[1] / 'src/satground/losses.py'),
                               'diagnostic_script': sha256(__file__)},
                torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                scipy_version=str(scipy.__version__), real_data_read=False, optimizer_updates=0,
                gpu_used=False, generator_improvement_established=False,
                loss_weights_are_fixed_from_fitting_protocol=True,
                limitations=[
                    'These maps are synthetic and do not establish improvement of a generator or an adapter.',
                    'Region and boundary probability gradients omit the frozen segmenter and generator Jacobians.',
                    'A zero boundary derivative on a uniform map does not imply zero region, RGB or perceptual corrective signal.',
                    'A displaced building has both missing and extra regions; its higher combined loss than an empty map does not establish generator collapse.',
                    'Hard-mask translations are discrete comparisons; only sigmoid-map translations have parameter derivatives.',
                    'Max/min pooling and absolute error have nondifferentiable ties; finite secants can differ at kinks.',
                    'Auxiliary hard-mask geometry includes separately counted crop-closing edges; it is not a training loss.',
                    'The synthetic support fraction is not a measured distribution of VIGOR labels.',
                    'No candidate loss, hyperparameter search, training or held-out evaluation is performed.',
                ])


def build_report():
    """Bound CPU work without leaving a caller's thread configuration changed."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    try:
        return _build_report()
    finally:
        torch.set_num_threads(old_threads)


def write_report(output):
    output = Path(output)
    require(output.suffix == '.json', 'Diagnostic output must be a JSON file.')
    require(not output.exists(), 'Use a fresh diagnostic path; earlier evidence is preserved.')
    require(not output.with_suffix(output.suffix + '.tmp').exists(), 'A partial diagnostic exists; preserve it and use a fresh path.')
    report = build_report()
    _require_finite(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents replacing an output made concurrently.
    import json
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = write_report(args.output)
    print(f"Measured {result['synthetic_case_count']} fixed CPU cases and {result['differentiable_probe_count']} parameter probes; no training updates.")
