"""Synthetic CPU completion/rejection tests; never mutate real pinned inputs."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import report_contour_objective_diagnosis as reporter
from satground.common import ResearchError, load_json, object_hash, save_json, sha256


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def payload(space_lengths, components, actual_names):
    result = {space: {name: torch.zeros(length) for name in components}
              for space, length in space_lengths}
    for name in actual_names:
        space = name.rsplit('_', 1)[1]
        result[name] = torch.zeros(dict(space_lengths)[space])
    return result


@pytest.fixture
def completed(tmp_path, monkeypatch):
    """Fixture paths are all inside tmp_path; global workspace inputs stay untouched."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA called'))
    root = tmp_path.resolve()
    monkeypatch.setattr(reporter, 'ROOT', root)
    bundle = root / 'data/synthetic-fitting8'
    (bundle / 'labels').mkdir(parents=True)
    rows = [dict(sample_id=f'SYNTHETIC-VIEW-{i + 1}') for i in range(8)]
    labels = []
    for i, row in enumerate(rows):
        building = np.zeros((256, 256), dtype=np.float32)
        building[32:96, 40 + i:104 + i] = 1
        valid = np.ones_like(building)
        np.savez(bundle / 'labels' / (row['sample_id'] + '.npz'), building=building, valid=valid)
        labels.append((building, valid))
    hp_path, cp_path = root / 'configs/protocol.json', root / 'configs/candidate-protocol.json'
    hp = dict(optimizer_updates=0, weights=dict(rgb=1., lpips=.1, region=.5, boundary=.5),
              image_gradient_relative_tolerance=5e-5, adapter_gradient_relative_tolerance=2e-4)
    cp = dict(optimizer_updates=0, historical_protocol='configs/protocol.json',
              weights=dict(rgb=1., lpips=.1, region=.5, surface=.5),
              required_supported_nonzero_finite_adapter_views=6,
              image_gradient_relative_tolerance=5e-5, adapter_gradient_relative_tolerance=2e-4)
    save_json(hp_path, hp)
    save_json(cp_path, cp)
    monkeypatch.setattr(reporter, 'HISTORICAL_PROTOCOL', hp_path)
    monkeypatch.setattr(reporter, 'CANDIDATE_PROTOCOL', cp_path)
    files = {str(p): sha256(p) for p in (hp_path, cp_path, *sorted((bundle / 'labels').iterdir()))}
    plan = [dict(state=name, fitting_view=i + 1, sample_id=row['sample_id'], seed=101 + i)
            for name in ('A0', 'A2', 'A3') for i, row in enumerate(rows)]
    hi, ci = dict(protocol=hp, files=files, plan=plan), dict(protocol=cp, files=files, plan=plan[:8])
    ho, co = root / 'runs/historical24', root / 'runs/candidate8'
    monkeypatch.setattr(reporter, 'HISTORICAL_OUTPUT', ho)
    monkeypatch.setattr(reporter, 'CANDIDATE_OUTPUT', co)
    for output, identity, module, count in ((ho, hi, reporter.historical, 24), (co, ci, reporter.candidate, 8)):
        state = dict(status='complete', current_check=None, expected=count, completed=count,
            optimizer_updates=0, parameters_unchanged=True, identity=identity,
            identity_hash=object_hash(identity), receipts=[])
        records = []
        for i in range(count):
            building, valid = labels[i % 8]
            record = dict(plan=identity['plan'][i], parameters_unchanged=True, optimizer_updates=0,
                          frozen_models_verified=True, gradient_sum_relative_error=dict(image=0., adapter=0.))
            if count == 24:
                vectors = payload((('image', 196608), ('adapter', 4448)), module.COMPONENTS,
                                  ('actual_image', 'actual_adapter'))
                _, counts = module.boundary_partition(torch.zeros(1, 1, 256, 256),
                    torch.from_numpy(building)[None, None], torch.from_numpy(valid)[None, None])
                record.update(support=counts, raw_components={k: 0. for k in module.COMPONENTS},
                              objective_values={k: 0. for k in ('rgb', 'lpips', 'region', 'boundary', 'total')})
            else:
                vectors = payload((('image', 196608), ('adapter', 4448)), module.COMPONENTS,
                    ('actual_A2_image', 'actual_A2_adapter', 'actual_combined_image', 'actual_combined_adapter'))
                prepared = reporter.prepare_surface_labels(building, valid)
                record.update(support=prepared['counts'], surface_provenance=prepared['provenance'],
                    raw_components={k: 0. for k in module.COMPONENTS},
                    A2_objective_values={k: 0. for k in ('rgb', 'lpips', 'region', 'boundary', 'total')},
                    A2_gradient_sum_relative_error=dict(image=0., adapter=0.),
                    historical_A2_gradient_relative_error=dict(image=0., adapter=0.))
            record.update({space + '_gradients': module.gradient_summary(vectors[space]) for space in ('image', 'adapter')})
            state['receipts'].append(module.commit_record(output, i + 1, record, vectors, state['identity_hash']))
            records.append(record)
        save_json(output / 'summary.json', module.aggregate(records, state))
        state['summary_sha256'] = sha256(output / 'summary.json')
        save_json(output / 'status.json', state)
    monkeypatch.setattr(reporter.historical, 'protocol_identity', lambda path: (hp, bundle, {}, rows, {}, hi))
    monkeypatch.setattr(reporter.candidate, 'protocol_identity', lambda path: (cp, bundle, {}, rows, ho, [], ci))
    monkeypatch.setattr(reporter, 'other_gpu_work', lambda: [])
    for owner, attribute, name in ((reporter.historical, 'LOCK', 'historical.lock'),
        (reporter.candidate, 'LOCK', 'candidate.lock'), (reporter, 'FITTING_LOCK', 'fitting.lock')):
        monkeypatch.setattr(owner, attribute, root / 'runs' / name)
    baseline = dict(synthetic_case_count=52, frozen_case_identity_sha256=reporter.compared.FROZEN_CASE_IDENTITY,
                    source_hashes={'synthetic_source': 'f' * 64})
    surface = dict(fixed_synthetic_case_count=52, fixed_case_identity_sha256=reporter.compared.FROZEN_CASE_IDENTITY,
        source_hashes={'synthetic_source': 'f' * 64}, prior_gate=dict(method=reporter.POLICY),
        training_clearance=False, harmful_far_combined_ordering=True,
        no_training_on_harmful_controlled_behavior=True, training_blockers=['Synthetic harmful far ordering.'],
        optimizer_updates=0, gpu_used=False, controlled_gate_passed=True)
    br, sr = root / 'reports/controlled.json', root / 'reports/surface.json'
    save_json(br, baseline)
    save_json(sr, surface)
    monkeypatch.setattr(reporter, 'CONTROLLED_REPORT', br)
    monkeypatch.setattr(reporter, 'SURFACE_REPORT', sr)
    monkeypatch.setattr(reporter.controlled, 'build_report', lambda: copy.deepcopy(baseline))
    monkeypatch.setattr(reporter.compared, 'build_comparison', lambda: copy.deepcopy(surface))
    return dict(root=root, ho=ho, co=co, bundle=bundle, baseline=baseline, surface=surface,
                output=root / 'runs/completion', rows=rows)


def rewrite_record(fixture, output, ordinal, change):
    """Tamper only explicit synthetic fixture files, updating its exact receipt."""
    directory = output / 'checks' / f'{ordinal:04d}'
    assert directory.resolve().is_relative_to(fixture['root'])
    record = load_json(directory / 'record.json')
    change(record)
    save_json(directory / 'record.json', record)
    receipt = load_json(directory / 'receipt.json')
    receipt['record_sha256'] = sha256(directory / 'record.json')
    save_json(directory / 'receipt.json', receipt)
    state = load_json(output / 'status.json')
    state['receipts'][ordinal - 1] = sha256(directory / 'receipt.json')
    save_json(output / 'status.json', state)


def test_complete32_reproduces_to_fresh_aggregate_only_paths(completed):
    first = reporter.report(completed['output'])
    second_path = completed['root'] / 'runs/reproduced'
    assert reporter.report(second_path) == first
    assert first['total_finite_checks'] == 32 and first['optimizer_updates'] == 0
    assert not first['training_clearance'] and not first['held_out_generalization_evidence']
    assert {p.name for p in completed['output'].iterdir()} == {
        'historical-aggregate.json', 'candidate-aggregate.json', 'completion-audit.json'}
    for path in completed['output'].iterdir():
        assert path.read_bytes() == (second_path / path.name).read_bytes()
        text = path.read_text()
        assert 'SYNTHETIC-VIEW' not in text and str(completed['root']) not in text
        assert 'sample_id' not in text and 'building_array_sha256' not in text
    with pytest.raises(ResearchError, match='fresh'):
        reporter.report(completed['output'])


@pytest.mark.parametrize('field,value', [('status', 'startup_complete'), ('completed', 7),
    ('optimizer_updates', 1), ('parameters_unchanged', False), ('current_check', 8)])
def test_incomplete_or_updated_status_is_rejected(completed, field, value):
    path = completed['co'] / 'status.json'
    state = load_json(path)
    state[field] = value
    save_json(path, state)
    with pytest.raises(ResearchError, match='incomplete|zero-update'):
        reporter.report(completed['output'])
    assert not completed['output'].exists()


def test_current_identity_and_explicit_synthetic_pin_are_checked(completed, monkeypatch):
    pin = completed['bundle'] / 'labels' / (completed['rows'][0]['sample_id'] + '.npz')
    assert pin.resolve().is_relative_to(completed['root'])
    pin.write_bytes(pin.read_bytes() + b'SYNTHETIC-TAMPER')
    with pytest.raises(ResearchError, match='Pinned'):
        reporter.report(completed['output'])


@pytest.mark.parametrize('kind', ['mask_support', 'mask_provenance', 'historical_comparison', 'frozen_reference'])
def test_authenticated_but_false_record_claims_are_rejected(completed, kind):
    def change(record):
        if kind == 'mask_support': record['support']['interface_source_pixels'] += 1
        elif kind == 'mask_provenance': record['surface_provenance']['building_array_sha256'] = '0' * 64
        elif kind == 'historical_comparison': record['historical_A2_gradient_relative_error']['adapter'] = .01
        else: record['frozen_models_verified'] = False
    output = completed['ho'] if kind == 'frozen_reference' else completed['co']
    rewrite_record(completed, output, 1, change)
    # Recreate only the synthetic stored aggregate, so the independent mask or
    # raw-gradient check (rather than merely a stale aggregate) must reject it.
    if kind != 'frozen_reference':
        state = load_json(output / 'status.json')
        records = reporter.candidate.verified_records(output, state)
        save_json(output / 'summary.json', reporter.candidate.aggregate(records, state))
        state['summary_sha256'] = sha256(output / 'summary.json')
        save_json(output / 'status.json', state)
    with pytest.raises(ResearchError, match='support|provenance|comparison|frozen'):
        reporter.report(completed['output'])


def test_raw_gradient_receipt_and_aggregate_tampering_are_rejected(completed):
    path = completed['co'] / 'checks/0001/gradients.pt'
    assert path.resolve().is_relative_to(completed['root'])
    path.write_bytes(path.read_bytes() + b'SYNTHETIC-TAMPER')
    with pytest.raises(ResearchError, match='hashes'):
        reporter.report(completed['output'])


@pytest.mark.parametrize('kind', ['identity', 'historical_status', 'summary', 'nonfinite_vector'])
def test_material_completion_claims_fail_closed(completed, kind):
    output = completed['ho'] if kind == 'historical_status' else completed['co']
    state = load_json(output / 'status.json')
    if kind == 'identity':
        state['identity']['protocol']['weights']['rgb'] = 2.
        state['identity_hash'] = object_hash(state['identity'])
    elif kind == 'historical_status':
        state['status'] = 'failed'
    elif kind == 'summary':
        summary = load_json(output / 'summary.json')
        summary['training_clearance'] = True
        save_json(output / 'summary.json', summary)
        state['summary_sha256'] = sha256(output / 'summary.json')
    else:
        directory = output / 'checks/0001'
        assert directory.resolve().is_relative_to(completed['root'])
        vectors = torch.load(directory / 'gradients.pt', map_location='cpu', weights_only=True)
        vectors['adapter']['surface'][0] = float('nan')
        torch.save(vectors, directory / 'gradients.pt')
        receipt = load_json(directory / 'receipt.json')
        receipt['vectors_sha256'] = sha256(directory / 'gradients.pt')
        save_json(directory / 'receipt.json', receipt)
        state['receipts'][0] = sha256(directory / 'receipt.json')
    save_json(output / 'status.json', state)
    with pytest.raises(ResearchError, match='identity|incomplete|aggregate|Invalid candidate'):
        reporter.report(completed['output'])
    assert not completed['output'].exists()


@pytest.mark.parametrize('failure', ['source', 'clearance'])
def test_controlled_reports_and_no_training_blocker_cannot_be_overridden(completed, failure):
    if failure == 'source':
        changed = copy.deepcopy(completed['baseline'])
        changed['source_hashes']['synthetic_source'] = '0' * 64
        save_json(reporter.CONTROLLED_REPORT, changed)
    else:
        completed['surface']['training_clearance'] = True
        save_json(reporter.SURFACE_REPORT, completed['surface'])
    with pytest.raises(ResearchError, match='reproducible|blocker'):
        reporter.report(completed['output'])


@pytest.mark.parametrize('kind', ['historical_lock', 'candidate_lock', 'fitting_lock', 'worker'])
def test_completion_requires_all_locks_and_workers_gone(completed, monkeypatch, kind):
    if kind == 'worker': monkeypatch.setattr(reporter, 'other_gpu_work', lambda: [123])
    else:
        path = {'historical_lock': reporter.historical.LOCK, 'candidate_lock': reporter.candidate.LOCK,
                'fitting_lock': reporter.FITTING_LOCK}[kind]
        assert path.resolve().is_relative_to(completed['root'])
        path.write_text('SYNTHETIC-LOCK')
    with pytest.raises(ResearchError, match='lock|worker'):
        reporter.report(completed['output'])


def test_outputs_must_remain_private(completed):
    with pytest.raises(ResearchError, match='private'):
        reporter.report(completed['root'] / 'reports/wrong')


def test_feasibility_rail_never_removes_the_controlled_training_blocker(completed):
    old = load_json(completed['ho'] / 'summary.json')
    new = load_json(completed['co'] / 'summary.json')
    new['feasibility_rail_passed'] = True
    new['supported_nonzero_finite_adapter_views'] = 8
    result = reporter.interpretation(old, new, completed['surface'])
    assert result['candidate_feasibility_rail_passed'] and result['training_clearance'] is False
