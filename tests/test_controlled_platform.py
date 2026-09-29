"""Input-fingerprint integrity tests; no research data, losses, model or GPU run.

Deliberately changed fingerprints below are synthetic corruption/scope tests,
not evidence that any platform changed the experiment's actual float64 bytes.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import diagnose_controlled_platform as diagnostic
from satground.common import ResearchError


REFERENCE = Path(__file__).parent / 'fixtures/controlled-platform-reference.json'


@pytest.fixture(scope='module', autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reference():
    # Only the committed, fully synthetic52 input definitions are read.
    return json.loads(REFERENCE.read_text(encoding='utf-8'))


def reidentify(value):
    value.update(diagnostic._identities(value['protocol'], value['cases']))
    return value


def test_committed_reference_preserves_exact_frozen_input_identity(reference):
    diagnostic.validate_fingerprint(reference)
    assert reference['case_count'] == len(reference['cases']) == 52
    assert len({row['name'] for row in reference['cases']}) == 52
    assert reference['raw_identity_sha256'] == diagnostic.FROZEN_WINDOWS_IDENTITY
    assert diagnostic._identities(reference['protocol'], reference['cases']) == {
        key: reference[key] for key in ('raw_identity_sha256', 'recipe_identity_sha256', 'canonical_identity_sha256')}
    assert reference['probability_quantization'] == diagnostic.QUANTIZATION
    assert reference['probability_quantization']['scope'] == 'probability arrays only; no targets, validity or recipe rounding'


def test_current_fingerprint_reproduces_reference_without_loss_data_or_gpu(reference, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA was queried'))
    monkeypatch.setattr(diagnostic, 'validate_inputs', diagnostic.validate_inputs)
    import contour_objective_cases
    monkeypatch.setattr(contour_objective_cases, 'measure_case', lambda *args: pytest.fail('A loss was evaluated'))
    monkeypatch.setattr(contour_objective_cases, 'build_report', lambda: pytest.fail('A loss report was evaluated'))
    threads = torch.get_num_threads()
    current = diagnostic.fingerprint()
    assert torch.get_num_threads() == threads == 1
    compared = diagnostic.compare_fingerprints(current, reference)
    assert compared['portable_input_check_passed'] and all(compared['checks'].values())
    assert compared['exact_raw_identity'] and compared['raw_probability_changed_count'] == 0
    assert current['raw_identity_sha256'] == diagnostic.FROZEN_WINDOWS_IDENTITY
    assert not current['gpu_used'] and not current['real_data_read'] and not current['losses_evaluated']
    assert current['optimizer_updates'] == 0 and not current['raw_identity_guard_changed']
    epsilon = 4 * np.finfo(np.float64).eps
    assert all(row['probability_grid_maximum_absolute_error'] <= .5 / diagnostic.GRID_SCALE + epsilon
               for row in current['cases'])


@pytest.mark.parametrize('field', ['parameters', 'shape', 'family', 'name', 'order', 'protocol',
                                  'target_sha256', 'validity_sha256', 'probability_grid_sha256'])
def test_portable_check_never_rounds_recipes_or_binary_labels(reference, field):
    current = copy.deepcopy(reference)
    if field == 'parameters': current['cases'][0]['parameters']['deliberate_exact_recipe_change'] = 1e-15
    elif field == 'shape': current['cases'][0]['shape'][0] += 1
    elif field == 'family': current['cases'][0]['family'] += '-deliberately-changed'
    elif field == 'name': current['cases'][0]['name'] += '-deliberately-changed'
    elif field == 'order': current['cases'][0], current['cases'][1] = current['cases'][1], current['cases'][0]
    elif field == 'protocol': current['protocol']['deliberate_exact_recipe_change'] = 1e-15
    else: current['cases'][0][field] = '0' * 64
    reidentify(current)  # An internally authenticated change must still fail comparison.
    result = diagnostic.compare_fingerprints(current, reference)
    assert not result['portable_input_check_passed'] and not result['experiment_identity_approved']
    if field in ('target_sha256', 'validity_sha256'):
        assert result['checks']['exact_probability_grid']
    if field == 'probability_grid_sha256':
        assert result['checks']['exact_targets'] and result['checks']['exact_validity']


def test_deliberate_raw_byte_change_is_only_a_grid_scope_check(reference):
    current = copy.deepcopy(reference)
    current['cases'][0]['probability_sha256'] = '0' * 64
    reidentify(current)
    result = diagnostic.compare_fingerprints(current, reference)
    assert result['portable_input_check_passed'] and not result['exact_raw_identity']
    assert result['raw_probability_changed_count'] == 1
    assert not result['experiment_identity_approved']
    assert result['shared_grid_per_pixel_difference_bound'] == 1. / diagnostic.GRID_SCALE + 4 * np.finfo(np.float64).eps
    assert 'does not establish bitwise experimental reproduction' in result['limitation']
    # The committed reference and live strict identity are not rewritten.
    assert json.loads(REFERENCE.read_text())['raw_identity_sha256'] == diagnostic.FROZEN_WINDOWS_IDENTITY


def test_checked_test_identity_accepts_only_exact_recipe_labels_and_grid(tmp_path, reference, monkeypatch):
    current = copy.deepcopy(reference)
    current['cases'][0]['probability_sha256'] = '0' * 64
    reidentify(current)
    monkeypatch.setattr(diagnostic, 'fingerprint', lambda: current)
    raw, result = diagnostic.checked_test_identity(REFERENCE)
    assert raw == current['raw_identity_sha256'] and not result['exact_raw_identity']
    assert result['portable_input_check_passed'] and not result['experiment_identity_approved']
    current['cases'][0]['probability_grid_sha256'] = '0' * 64
    reidentify(current)
    with pytest.raises(ResearchError, match='grid'):
        diagnostic.checked_test_identity(REFERENCE)
    invalid_reference = copy.deepcopy(reference)
    invalid_reference['raw_identity_sha256'] = '0' * 64
    path = tmp_path / 'invalid-reference.json'
    path.write_text(json.dumps(invalid_reference))
    monkeypatch.setattr(diagnostic, 'fingerprint', lambda: pytest.fail('A rejected reference triggered calculation'))
    with pytest.raises(ResearchError, match='original frozen Windows identity'):
        diagnostic.checked_test_identity(path)


@pytest.mark.parametrize('failure', ['format', 'grid', 'count', 'duplicate', 'identity', 'missing_definition',
                                   'real_data', 'gpu', 'updates', 'raw_guard', 'losses'])
def test_malformed_or_tampered_fingerprints_are_rejected(reference, failure):
    current = copy.deepcopy(reference)
    if failure == 'format': current['format'] = 'unknown'
    elif failure == 'grid': current['probability_quantization']['scale'] = 10 ** 10
    elif failure == 'count': current['cases'].pop()
    elif failure == 'duplicate': current['cases'][1]['name'] = current['cases'][0]['name']
    elif failure == 'identity': current['cases'][0]['target_sha256'] = '0' * 64
    elif failure == 'missing_definition': del current['cases'][0]['validity_sha256']
    elif failure == 'real_data': current['real_data_read'] = True
    elif failure == 'gpu': current['gpu_used'] = True
    elif failure == 'updates': current['optimizer_updates'] = 1
    elif failure == 'raw_guard': current['raw_identity_guard_changed'] = True
    else: current['losses_evaluated'] = True
    with pytest.raises((ResearchError, KeyError)):
        diagnostic.validate_fingerprint(current)


def test_fresh_cli_output_is_input_only_and_never_overwrites_evidence(tmp_path, reference, monkeypatch):
    monkeypatch.setattr(diagnostic, 'fingerprint', lambda: copy.deepcopy(reference))
    path = tmp_path / 'fingerprint.json'
    monkeypatch.setattr(sys, 'argv', ['diagnose_controlled_platform.py', '--output', str(path), '--reference', str(REFERENCE)])
    diagnostic.main()
    result = json.loads(path.read_text())
    assert result['reference_comparison']['portable_input_check_passed']
    assert not result['reference_comparison']['experiment_identity_approved']
    original = path.read_bytes()
    with pytest.raises(ResearchError, match='fresh'):
        diagnostic.main()
    assert path.read_bytes() == original
    rejected = tmp_path / 'wrong.txt'
    monkeypatch.setattr(sys, 'argv', ['diagnose_controlled_platform.py', '--output', str(rejected)])
    with pytest.raises(ResearchError, match='JSON'):
        diagnostic.main()
    assert not rejected.exists()


def test_cli_requested_raw_identity_cannot_be_bypassed_by_grid_agreement(tmp_path, reference, monkeypatch):
    current = copy.deepcopy(reference)
    current['cases'][0]['probability_sha256'] = '0' * 64
    reidentify(current)
    assert diagnostic.compare_fingerprints(current, reference)['portable_input_check_passed']
    monkeypatch.setattr(diagnostic, 'fingerprint', lambda: current)
    output = tmp_path / 'rejected.json'
    monkeypatch.setattr(sys, 'argv', ['diagnose_controlled_platform.py', '--output', str(output),
        '--expect-raw-identity', diagnostic.FROZEN_WINDOWS_IDENTITY])
    with pytest.raises(ResearchError, match='raw identity'):
        diagnostic.main()
    assert not output.exists()
