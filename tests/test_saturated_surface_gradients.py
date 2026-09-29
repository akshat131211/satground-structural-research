"""New saturated-v2 probe synthetic CPU tests; no real data, model or GPU jobs."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import probe_saturated_surface_gradients as probe
from satground.common import ResearchError, object_hash, save_json, sha256
from saturated_surface_objective import prepare_surface_labels
from test_surface_candidate_gradients import SyntheticObjective, ledger as v1_ledger, synthetic_check as v1_check


def protocol():
    p = v1_ledger()['identity']['protocol'].copy()
    p.update(study='reviewed_saturated_surface_feasibility', surface_policy='valid_surface_saturated_v2',
        completed_failed_candidate_probe='runs/reviewed-surface-candidate8',
        controlled_report='reports/saturated-surface-diagnosis/controlled-cases.json',
        documentation='docs/VALID_SURFACE_SATURATION_REVISION.md')
    return p


@pytest.mark.parametrize('key,value', [('expected_checks', 9), ('states', ['A2']), ('optimizer_updates', 1),
    ('radius', 9), ('surface_policy', 'valid_surface_excess_v1'), ('view_seed_base', 102),
    ('required_supported_nonzero_finite_adapter_views', 5), ('parameter_changes_allowed', True)])
def test_saturated_protocol_is_fixed(key, value):
    p = protocol(); probe.validate_protocol(p)
    p[key] = value
    with pytest.raises(ResearchError, match='protocol'): probe.validate_protocol(p)


@pytest.mark.parametrize('has_interface', [True, False])
def test_saturated_measure_uses_original_render_graph_and_finite_gradients(has_interface):
    y = np.zeros((12, 12), np.float32)
    if has_interface: y[4:8, 4:8] = 1
    v = np.ones_like(y)
    prepared = prepare_surface_labels(y, v)
    assert 'near_band_pixels' in prepared['counts'] and 'band_pixels' not in prepared['counts']
    labels = dict(building=torch.from_numpy(y)[None, None], valid=torch.from_numpy(v)[None, None])
    parameter = torch.nn.Parameter(torch.tensor([.3, -.2]))
    before = parameter.detach().clone()
    base = torch.linspace(0, 1, 3 * 12 * 12).reshape(1, 3, 12, 12)
    prediction = parameter[0] * base + parameter[1].square()
    target = torch.zeros_like(prediction)
    objective = SyntheticObjective()
    _, values = objective(prediction, target, labels)
    historical = dict(record=dict(raw_components={k: values[k] for k in ('rgb', 'lpips', 'region')}),
        vectors={space: {k: torch.zeros(length) for k in ('rgb', 'lpips', 'region')}
                 for space, length in (('image', prediction.numel()), ('adapter', 2))})
    record, vectors = probe.measure(objective, prediction, target, labels, prepared, (parameter,), protocol(), historical)
    assert torch.equal(parameter, before) and parameter.grad is None
    assert (record['adapter_gradients']['weighted_norms']['surface'] > 0) == has_interface
    assert all(value < 2e-4 for value in record['gradient_sum_relative_error'].values())
    assert all(torch.isfinite(v).all() for v in vectors['adapter'].values())
    assert record['surface_provenance']['policy'] == 'valid_surface_saturated_v2'


def ledger():
    state = v1_ledger()
    state['identity']['protocol'] = protocol()
    state['identity_hash'] = object_hash(state['identity'])
    return state


def check(state, ordinal, nonzero=True):
    record, vectors = v1_check(state, ordinal, nonzero)
    record['surface_provenance']['policy'] = 'valid_surface_saturated_v2'
    s = record['support']
    s['near_band_pixels'] = s.pop('band_pixels')
    s['near_band_class_pixels'] = s.pop('class_band_pixels')
    s.update(surface_scored_pixels=80 if nonzero else 0, saturated_distance_pixels=65 if nonzero else 0,
        beyond_radius_saturated_pixels=60 if nonzero else 0,
        class_scored_pixels=[34, 46] if nonzero else [0, 0], class_weight_mass=[30., 42.] if nonzero else [0., 0.],
        class_denominators=[32., 42.] if nonzero else [32., 32.])
    return record, vectors


def add_check(output, state, ordinal, nonzero=True):
    record, vectors = check(state, ordinal, nonzero)
    state['receipts'].append(probe.previous.commit_record(output, ordinal, record, vectors, state['identity_hash']))
    state['completed'] = ordinal


def test_exact_one_startup_plus_seven_and_unambiguous_public_support(tmp_path):
    state = ledger(); add_check(tmp_path, state, 1)
    assert len(probe.verified_records(tmp_path, state)) == 1
    for i in range(2, 9): add_check(tmp_path, state, i, nonzero=i != 7)
    report = probe.aggregate(probe.verified_records(tmp_path, state), state)
    assert report['total_checks'] == 8 and report['supported_nonzero_finite_adapter_views'] == 7
    assert report['feasibility_rail_passed'] and report['controlled_prerequisite_clearance']
    assert not report['training_clearance'] and not report['method_selected'] and not report['server_gate_eligible']
    assert report['support']['near_band_pixels'] == 140
    assert report['support']['surface_scored_pixels'] == 560
    assert report['support']['saturated_distance_pixels'] == 455
    assert report['support']['class_weight_mass'] == [210., 294.]
    assert 'band_pixels' not in report['support'] and 'synthetic-private-id' not in str(report)
    with pytest.raises(ResearchError, match='Partial'): add_check(tmp_path, state, 8)


def test_five_nonzero_gradients_do_not_pass_feasibility(tmp_path):
    state = ledger()
    for i in range(1, 9): add_check(tmp_path, state, i, nonzero=i <= 5)
    report = probe.aggregate(probe.verified_records(tmp_path, state), state)
    assert not report['feasibility_rail_passed'] and not report['training_clearance']


@pytest.mark.parametrize('file', ['record.json', 'gradients.pt', 'receipt.json'])
def test_immutable_saturated_receipts_detect_tampering(tmp_path, file):
    state = ledger(); add_check(tmp_path, state, 1)
    path = tmp_path / 'checks' / '0001' / file
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ResearchError): probe.verified_records(tmp_path, state)


@pytest.mark.parametrize('change', ['old_policy', 'ambiguous_band', 'updated_parameters', 'optimizer_step', 'nonfinite'])
def test_rehashed_wrong_policy_or_update_still_fails(tmp_path, change):
    state = ledger(); record, vectors = check(state, 1)
    if change == 'old_policy': record['surface_provenance']['policy'] = 'valid_surface_excess_v1'
    elif change == 'ambiguous_band': record['support']['band_pixels'] = 20
    elif change == 'updated_parameters': record['parameters_unchanged'] = False
    elif change == 'optimizer_step': record['optimizer_updates'] = 1
    else: vectors['adapter']['surface'][0] = float('nan')
    state['receipts'].append(probe.previous.commit_record(tmp_path, 1, record, vectors, state['identity_hash']))
    state['completed'] = 1
    with pytest.raises(ResearchError): probe.verified_records(tmp_path, state)


def test_partial_check_and_incomplete_aggregate_fail_closed(tmp_path):
    state = ledger(); add_check(tmp_path, state, 1)
    record, _ = check(state, 1)
    with pytest.raises(ResearchError, match='eight'): probe.aggregate([record], state)
    (tmp_path / 'checks' / '0002').mkdir()
    with pytest.raises(ResearchError, match='partial'): probe.verified_records(tmp_path, state)


def test_failed_controlled_case_blocks_even_if_reproduction_matches(tmp_path):
    path = tmp_path / 'controlled.json'; report = dict(training_clearance=False)
    save_json(path, report)
    with pytest.raises(ResearchError, match='clearance'): probe.validate_controlled(path, lambda: report)


def test_controlled_report_requires_exact_fresh_reproduction(tmp_path):
    path = tmp_path / 'controlled.json'; report = dict(training_clearance=True, fixed_case_count=52)
    save_json(path, report)
    assert probe.validate_controlled(path, lambda: report) == report
    with pytest.raises(ResearchError, match='reproduce'):
        probe.validate_controlled(path, lambda: dict(training_clearance=True, fixed_case_count=51))


def test_prior_eight_receipts_and_summary_reverified_without_running_them(tmp_path, monkeypatch):
    identity = {'synthetic': True}; report = {'checks': 8}
    save_json(tmp_path / 'summary.json', report)
    state = dict(status='complete', completed=8, expected=8, identity=identity,
        identity_hash=object_hash(identity), summary_sha256=sha256(tmp_path / 'summary.json'))
    save_json(tmp_path / 'status.json', state)
    calls = []
    monkeypatch.setattr(probe.previous, 'verified_records', lambda output, state: calls.append('receipts') or ['synthetic'])
    monkeypatch.setattr(probe.previous, 'aggregate', lambda records, state: report)
    assert probe.validate_previous(tmp_path, identity)[1] == ['synthetic'] and calls == ['receipts']
    save_json(tmp_path / 'summary.json', {'changed': True})
    with pytest.raises(ResearchError, match='aggregate'): probe.validate_previous(tmp_path, identity)


def test_prior_probe_must_be_complete(tmp_path):
    identity = {'synthetic': True}
    save_json(tmp_path / 'status.json', dict(status='startup_complete', completed=1, expected=8,
        identity=identity, identity_hash=object_hash(identity)))
    with pytest.raises(ResearchError, match='incomplete'): probe.validate_previous(tmp_path, identity)


@pytest.mark.parametrize('change', ['wrong_total', 'wrong_normalizer', 'old_band', 'missing_class'])
def test_support_bookkeeping_cannot_silently_relabel_the_old_band(change):
    record, _ = check(ledger(), 1)
    support = record['support']
    if change == 'wrong_total': support['surface_scored_pixels'] = 19
    elif change == 'wrong_normalizer': support['class_denominators'][0] = 30.
    elif change == 'old_band': support['band_pixels'] = 80
    else: support.pop('class_scored_pixels')
    with pytest.raises(ResearchError): probe.validate_support(support)
