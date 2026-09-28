"""Synthetic CPU candidate measurement and immutable receipt tests; no GPU runs."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import probe_surface_candidate_gradients as probe
from satground.common import ResearchError, load_json, object_hash, save_json, sha256
from satground.losses import structural_losses
from valid_surface_objective import prepare_surface_labels


def protocol():
    return dict(study='reviewed_valid_surface_feasibility',
        historical_protocol='configs/contour-objective-diagnosis/protocol.json',
        completed_historical_probe='runs/reviewed-contour-objective-diagnosis24',
        bundle='data/review/reviewed-fitting8-v1', states=['A0'], views=8, expected_checks=8,
        optimizer_updates=0, initialization_seed=17, view_seed_base=101,
        weights=dict(rgb=1., lpips=.1, region=.5, surface=.5),
        surface_policy='valid_surface_excess_v1', radius=8, class_mass_floor=32,
        required_supported_nonzero_finite_adapter_views=6,
        precision='float32_with_tf32_disabled_for_diagnostic_only',
        image_gradient_relative_tolerance=5e-5, adapter_gradient_relative_tolerance=2e-4,
        parameter_changes_allowed=False, target_label_changes_allowed=False,
        server_gate_eligible=False, held_out_generalization_evidence=False)


@pytest.mark.parametrize('key,value', [('expected_checks', 9), ('states', ['A2']), ('optimizer_updates', 1),
    ('radius', 9), ('class_mass_floor', 31), ('view_seed_base', 102), ('required_supported_nonzero_finite_adapter_views', 5),
    ('weights', dict(rgb=1, lpips=.1, region=.5, surface=.1)), ('parameter_changes_allowed', True)])
def test_different_protocol_fails_closed(key, value):
    p = protocol(); probe.validate_protocol(p)
    p[key] = value
    with pytest.raises(ResearchError, match='protocol'): probe.validate_protocol(p)


class SyntheticObjective:
    class Segmenter:
        @staticmethod
        def probabilities(image):
            p = image.mean(dim=1, keepdim=True).sigmoid()
            return torch.cat((p * 0, p * 0, p), dim=1)
    segmenter = Segmenter()

    @staticmethod
    def perceptual(image, target):
        return (image - target).square().mean().reshape(1)

    def __call__(self, prediction, target, labels):
        p = self.segmenter.probabilities(prediction)[:, 2:3]
        region, boundary = structural_losses(p, labels['building'], labels['valid'])
        rgb = F.l1_loss(prediction, target)
        perceptual = self.perceptual(prediction * 2 - 1, target * 2 - 1).mean()
        total = rgb + .1 * perceptual + .5 * region
        return total, {k: float(v.detach()) for k, v in dict(rgb=rgb, lpips=perceptual,
                       region=region, boundary=boundary, total=total).items()}


@pytest.mark.parametrize('has_interface', [True, False])
def test_direct_measurement_retains_render_graph_without_updates(has_interface):
    y = np.zeros((12, 12), np.float32)
    if has_interface: y[4:8, 4:8] = 1
    v = np.ones_like(y)
    prepared = prepare_surface_labels(y, v)
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
    assert record['adapter_gradients']['weighted_norms']['A2'] > 0
    assert (record['adapter_gradients']['weighted_norms']['surface'] > 0) == has_interface
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in vectors['adapter'].values())
    assert all(value < 2e-4 for value in record['gradient_sum_relative_error'].values())


def synthetic_check(state, ordinal, nonzero=True):
    vectors = dict(image={}, adapter={})
    for space, length in (('image', 3 * 256 * 256), ('adapter', 4448)):
        for i, key in enumerate(probe.COMPONENTS, 1):
            v = torch.zeros(length); v[0] = i if key != 'surface' or nonzero else 0
            vectors[space][key] = v
        reference = sum(vectors[space][k] for k in ('rgb', 'lpips', 'region'))
        vectors[f'actual_A2_{space}'] = reference
        vectors[f'actual_combined_{space}'] = reference + vectors[space]['surface']
    record = dict(plan=state['identity']['plan'][ordinal - 1], parameters_unchanged=True,
        optimizer_updates=0, frozen_models_verified=True, raw_components=dict(rgb=1., lpips=2., region=3., surface=4.),
        A2_objective_values=dict(rgb=1., lpips=2., region=3., total=2.7),
        surface_provenance=dict(policy='valid_surface_excess_v1', radius=8, mass_floor=32, coefficient=.5),
        support=dict(accepted_valid_pixels=100, domain_pixels=80, historical_positive_contour_pixels=12,
            interface_source_pixels=8 if nonzero else 0, historical_contour_pixels_retained_as_sources=8 if nonzero else 0,
            historical_contour_pixels_not_sources=4 if nonzero else 12, band_pixels=20 if nonzero else 0,
            domain_components=2, domain_components_without_observed_interface=1 if nonzero else 2,
            class_weight_mass=[12., 15.] if nonzero else [0., 0.], class_band_pixels=[9, 11] if nonzero else [0, 0],
            surface_support_present=nonzero),
        gradient_sum_relative_error=dict(image=0., adapter=0.), A2_gradient_sum_relative_error=dict(image=0., adapter=0.),
        historical_A2_gradient_relative_error=dict(image=1e-6, adapter=2e-6),
        **{space + '_gradients': probe.gradient_summary(vectors[space]) for space in ('image', 'adapter')})
    return record, vectors


def ledger():
    identity = dict(protocol=protocol(), plan=[dict(state='A0', fitting_view=i + 1,
                    sample_id=f'synthetic-private-id-{i + 1}', seed=101 + i) for i in range(8)])
    return dict(identity=identity, identity_hash=object_hash(identity), expected=8, completed=0, receipts=[])


def add_check(output, state, ordinal, nonzero=True):
    record, vectors = synthetic_check(state, ordinal, nonzero)
    state['receipts'].append(probe.commit_record(output, ordinal, record, vectors, state['identity_hash']))
    state['completed'] = ordinal


def test_one_startup_receipt_plus_seven_is_exactly_eight_not_nine(tmp_path):
    state = ledger()
    add_check(tmp_path, state, 1)
    assert len(probe.verified_records(tmp_path, state)) == 1
    for i in range(2, 9): add_check(tmp_path, state, i, nonzero=i != 7)
    records = probe.verified_records(tmp_path, state)
    report = probe.aggregate(records, state)
    assert report['total_checks'] == 8 and report['supported_nonzero_finite_adapter_views'] == 7
    assert report['feasibility_rail_passed'] and not report['method_selected'] and not report['training_clearance']
    assert report['feasibility_rail_is_not_training_clearance']
    assert not report['server_gate_eligible'] and not report['held_out_generalization_evidence']
    assert 'synthetic-private-id' not in str(report) and 'plan' not in report
    assert report['unsupported_views'] == 1
    with pytest.raises(ResearchError, match='Partial'): add_check(tmp_path, state, 8)


def test_feasibility_fails_if_only_five_have_candidate_gradients(tmp_path):
    state = ledger()
    for i in range(1, 9): add_check(tmp_path, state, i, nonzero=i <= 5)
    report = probe.aggregate(probe.verified_records(tmp_path, state), state)
    assert report['supported_nonzero_finite_adapter_views'] == 5 and not report['feasibility_rail_passed']
    assert not report['training_clearance']


@pytest.mark.parametrize('file', ['record.json', 'gradients.pt', 'receipt.json'])
def test_tampering_a_committed_check_fails_closed(tmp_path, file):
    state = ledger(); add_check(tmp_path, state, 1)
    path = tmp_path / 'checks' / '0001' / file
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ResearchError): probe.verified_records(tmp_path, state)


def test_partial_check_or_wrong_schedule_requires_preservation(tmp_path):
    state = ledger(); add_check(tmp_path, state, 1)
    (tmp_path / 'checks' / '0002').mkdir()
    with pytest.raises(ResearchError, match='partial'): probe.verified_records(tmp_path, state)


def test_rehashed_optimizer_update_cannot_become_a_valid_record(tmp_path):
    state = ledger()
    record, vectors = synthetic_check(state, 1)
    record['optimizer_updates'] = 1
    state['receipts'].append(probe.commit_record(tmp_path, 1, record, vectors, state['identity_hash']))
    state['completed'] = 1
    with pytest.raises(ResearchError, match='zero-update'): probe.verified_records(tmp_path, state)


def test_aggregate_requires_all_eight_checks():
    state = ledger(); state['completed'] = 1
    record, _ = synthetic_check(state, 1)
    with pytest.raises(ResearchError, match='eight'): probe.aggregate([record], state)


def test_historical_probe_must_be_completed(tmp_path):
    identity = {'synthetic': True}
    save_json(tmp_path / 'status.json', dict(status='startup_complete', completed=1, expected=24,
        identity=identity, identity_hash=object_hash(identity)))
    with pytest.raises(ResearchError, match='incomplete'): probe.validate_historical(tmp_path, identity)


def test_historical_summary_and_receipts_are_reverified(tmp_path, monkeypatch):
    identity = {'synthetic': True}
    report = {'checks': 24}
    save_json(tmp_path / 'summary.json', report)
    state = dict(status='complete', completed=24, expected=24, identity=identity,
                 identity_hash=object_hash(identity), summary_sha256=sha256(tmp_path / 'summary.json'))
    save_json(tmp_path / 'status.json', state)
    called = []
    monkeypatch.setattr(probe, 'historical_records', lambda output, state: called.append('receipts') or ['synthetic'])
    monkeypatch.setattr(probe, 'historical_aggregate', lambda records, state: report)
    assert probe.validate_historical(tmp_path, identity)[1] == ['synthetic']
    assert called == ['receipts']
    save_json(tmp_path / 'summary.json', {'changed': True})
    with pytest.raises(ResearchError, match='aggregate'): probe.validate_historical(tmp_path, identity)


def test_missing_components_and_nonfinite_gradients_fail_closed():
    with pytest.raises(ResearchError): probe.gradient_summary({'surface': torch.zeros(2)})
    vectors = {k: torch.ones(2) for k in probe.COMPONENTS}
    vectors['surface'][0] = float('nan')
    with pytest.raises(ResearchError): probe.gradient_summary(vectors)
