"""Fingerprint only the fixed synthetic inputs; no loss, model, data or GPU run.

The raw float64 identity remains the experiment's strict identity. A separate
1e-12 probability grid is solely a portable CPU-test check; target/validity
arrays, case order, shapes, parameters and the complete recipe remain exact.
This command never changes or bypasses an experimental comparator guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
import torch

from contour_objective_cases import PROTOCOL, fixed_cases, validate_inputs
from satground.common import ResearchError, object_hash


GRID_SCALE = 10 ** 12
FORMAT = 'controlled-platform-fingerprint-v1'
FROZEN_WINDOWS_IDENTITY = '724d8fb9e94ef38674c1baa22a328f78e344a8dc477ab7adc36a6f2876edac77'
QUANTIZATION = dict(scale=GRID_SCALE, rounding='numpy.rint ties-to-even',
                    byte_dtype='little-endian signed int64',
                    maximum_single_array_error=0.5 / GRID_SCALE,
                    scope='probability arrays only; no targets, validity or recipe rounding')
DEFINITION_KEYS = ('name', 'family', 'parameters', 'shape', 'probability_sha256',
                   'target_sha256', 'validity_sha256')
RECIPE_KEYS = ('name', 'family', 'parameters', 'shape')


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def _array_hash(array, dtype):
    return hashlib.sha256(np.asarray(array).astype(dtype).tobytes(order='C')).hexdigest()


def _definitions(cases, probability_key='probability_sha256'):
    return [{key: row[key] if key != 'probability_sha256' else row[probability_key]
             for key in DEFINITION_KEYS} for row in cases]


def _identities(protocol, cases):
    return dict(
        raw_identity_sha256=object_hash(dict(protocol=protocol, cases=_definitions(cases))),
        recipe_identity_sha256=object_hash(dict(protocol=protocol,
                                               cases=[{key: row[key] for key in RECIPE_KEYS} for row in cases])),
        canonical_identity_sha256=object_hash(dict(protocol=protocol,
                                                   cases=_definitions(cases, 'probability_grid_sha256'))),
    )


def _fingerprint():
    """Build input-only fingerprints of all52 unchanged CPU fixtures."""
    rows = []
    for case in fixed_cases():
        probability, target, valid = validate_inputs(case['probability'], case['target'], case['valid'])
        array = probability.detach().numpy()
        quantized = np.rint(array * GRID_SCALE).astype('<i8')
        restored = quantized.astype(np.float64) / GRID_SCALE
        rows.append(dict(
            name=case['name'], family=case['family'], parameters=case['parameters'],
            shape=list(target.shape), probability_sha256=_array_hash(array, '<f8'),
            target_sha256=_array_hash(target.detach().numpy(), '<f8'),
            validity_sha256=_array_hash(valid.detach().numpy(), '<f8'),
            probability_grid_sha256=_array_hash(quantized, '<i8'),
            probability_minimum=float(array.min()), probability_maximum=float(array.max()),
            probability_grid_maximum_absolute_error=float(np.max(np.abs(array - restored))),
        ))
    require(len(rows) == 52 and len({row['name'] for row in rows}) == 52,
            'The fixed synthetic set must contain exactly52 unique cases.')
    result = dict(format=FORMAT, protocol=PROTOCOL, probability_quantization=QUANTIZATION,
                  case_count=len(rows), cases=rows, **_identities(PROTOCOL, rows),
                  versions=dict(python=sys.version, platform=platform.platform(),
                                machine=platform.machine(), torch=str(torch.__version__),
                                numpy=str(np.__version__), scipy=str(scipy.__version__)),
                  raw_identity_guard_changed=False, real_data_read=False,
                  losses_evaluated=False, optimizer_updates=0, gpu_used=False)
    validate_fingerprint(result)
    return result


def fingerprint():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(PROTOCOL['cpu_threads'])
    try:
        return _fingerprint()
    finally:
        torch.set_num_threads(old_threads)


def validate_fingerprint(value):
    require(value.get('format') == FORMAT, 'Unsupported synthetic fingerprint format.')
    require(value.get('probability_quantization') == QUANTIZATION,
            'Probability precision policy differs from the declared1e-12 grid.')
    cases = value.get('cases', [])
    require(value.get('case_count') == len(cases) == 52,
            'The synthetic fingerprint must contain exactly52 cases.')
    require(len({row['name'] for row in cases}) == 52, 'Duplicate synthetic fingerprint names.')
    require(all(value.get(key) == digest for key, digest in _identities(value['protocol'], cases).items()),
            'Stored synthetic fingerprint identities do not reproduce.')
    require(not value.get('real_data_read') and not value.get('gpu_used')
            and value.get('optimizer_updates') == 0 and not value.get('raw_identity_guard_changed'),
            'A portability fingerprint must be input-only with unchanged raw guards.')


def compare_fingerprints(current, reference):
    """Return exact/portable comparisons without granting experiment clearance."""
    validate_fingerprint(current)
    validate_fingerprint(reference)
    rows = []
    for now, before in zip(current['cases'], reference['cases']):
        rows.append(dict(
            name=now['name'],
            exact_recipe=all(now[key] == before[key] for key in RECIPE_KEYS),
            exact_target=now['target_sha256'] == before['target_sha256'],
            exact_validity=now['validity_sha256'] == before['validity_sha256'],
            exact_raw_probability=now['probability_sha256'] == before['probability_sha256'],
            exact_probability_grid=now['probability_grid_sha256'] == before['probability_grid_sha256'],
        ))
    checks = dict(exact_protocol=current['protocol'] == reference['protocol'],
                  exact_recipes=all(row['exact_recipe'] for row in rows),
                  exact_targets=all(row['exact_target'] for row in rows),
                  exact_validity=all(row['exact_validity'] for row in rows),
                  exact_probability_grid=all(row['exact_probability_grid'] for row in rows))
    return dict(checks=checks, portable_input_check_passed=all(checks.values()),
                exact_raw_identity=current['raw_identity_sha256'] == reference['raw_identity_sha256'],
                raw_probability_changed_count=sum(not row['exact_raw_probability'] for row in rows),
                cases=rows, experiment_identity_approved=False,
                shared_grid_per_pixel_difference_bound=1. / GRID_SCALE + 4 * np.finfo(np.float64).eps,
                limitation='A shared rounded grid bounds differing original pixels within1e-12 plus float64 rounding; '
                           'it does not establish bitwise experimental reproduction or equivalent loss values.')


def checked_test_identity(reference_path):
    """Allow tests to inspect current inputs only after exact recipe/mask checks.

    Callers must retain this explicit test-only scope. Experimental comparators
    and their live raw-identity constants are deliberately untouched.
    """
    reference = json.loads(Path(reference_path).read_text(encoding='utf-8'))
    require(reference.get('raw_identity_sha256') == FROZEN_WINDOWS_IDENTITY,
            'The CPU-test reference must preserve the original frozen Windows identity.')
    current = fingerprint()
    comparison = compare_fingerprints(current, reference)
    require(comparison['portable_input_check_passed'],
            'Portable synthetic inputs differ in recipe, binary labels or declared probability grid.')
    return current['raw_identity_sha256'], comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--expect-raw-identity')
    args = parser.parse_args()
    result = fingerprint()
    if args.expect_raw_identity:
        require(result['raw_identity_sha256'] == args.expect_raw_identity,
                'The measured raw identity differs from the requested frozen identity.')
    if args.reference:
        result['reference_comparison'] = compare_fingerprints(
            result, json.loads(args.reference.read_text(encoding='utf-8')))
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        require(args.output.suffix == '.json', 'Use a JSON fingerprint output.')
        require(not args.output.exists(), 'Preserve the existing fingerprint; use a fresh path.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
