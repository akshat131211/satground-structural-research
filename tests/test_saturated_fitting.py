"""Synthetic CPU contract/recovery tests; no real masks, CUDA or GPU training."""
import copy
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import prepare_saturated_fitting as preparation
import run_saturated_fitting as runner
import train_saturated_fitting as trainer
from satground.adapter import StructuralAdapter
from satground.common import ResearchError, load_json, object_hash, read_jsonl, save_json, sha256, write_jsonl
from satground.losses import Objective


@pytest.fixture(scope='module', autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixed_protocol():
    return dict(study='reviewed_saturated_surface_fitting', original_bundle='data/review/reviewed-fitting8-v1',
        completed_feasibility='runs/reviewed-saturated-surface8',
        controlled_report='reports/saturated-surface-diagnosis/controlled-cases.json',
        steps=200, seed=17, render_seed=101, learning_rate=.00003, weight_decay=.0001, size=256,
        adapter_width=16, chunk_rows=4, checkpoint_rays=True, amp=False, accumulation=8, save_every=50,
        perceptual_weight=.1, region_weight=.5, boundary_weight=0., a3_surface_weight=.5,
        surface_policy='valid_surface_saturated_v2', radius=8, class_mass_floor=32,
        training_segmenter_revision='21b3847fae21ddee674abd31129307b6a1235bd9',
        evaluation_segmenter_revision='ec86afeba68e656629ccf47e0c8d2902f964917b',
        decision=dict(minimum_relative_boundary_reduction=.1, minimum_relative_nonperimeter_reduction=.1,
            minimum_iou_change=-.01, maximum_relative_lpips_increase=.05, minimum_fraction_views_jointly_improved=.5),
        startup_resume_updates_included_in_budget=1, main_study_eligible=False,
        server_gate_eligible=False, held_out_generalization_evidence=False)


@pytest.fixture
def trial_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA called'))
    root = tmp_path.resolve()
    protocol = fixed_protocol()
    bundle = root / protocol['original_bundle']
    trial = root / 'data/review/saturated-fitting8-v2'
    proof = root / protocol['completed_feasibility']
    (bundle / 'labels').mkdir(parents=True)
    rows = [dict(sample_id=f'SYNTHETIC-{i}') for i in range(8)]
    write_jsonl(bundle / 'manifest.jsonl', rows)
    save_json(bundle / 'manual-index.json', {row['sample_id']: dict(synthetic=True) for row in rows})
    originals, records = [], []
    for row in rows:
        building = np.zeros((256, 256), np.float32)
        building[60:140, 50:150] = 1
        valid = np.ones_like(building)
        file = bundle / 'labels' / (row['sample_id'] + '.npz')
        np.savez_compressed(file, building=building, valid=valid)
        originals.append((file, file.read_bytes()))
        prepared = preparation.prepare_surface_labels(building, valid)
        records.append(dict(support=prepared['counts'], surface_provenance=prepared['provenance']))
    for name in ('A0', 'A2', 'A3'):
        cfg = {k: protocol[k] for k in ('steps', 'seed', 'render_seed', 'learning_rate', 'weight_decay', 'size',
            'adapter_width', 'chunk_rows', 'checkpoint_rays', 'amp', 'accumulation', 'save_every', 'perceptual_weight',
            'training_segmenter_revision')}
        cfg.update(experiment=name, boundary_mode='legacy', region_weight=0. if name == 'A0' else .5,
            boundary_weight=.5 if name == 'A3' else 0., fitting_bundle=str(bundle),
            fitting_manifest=str(bundle / 'manifest.jsonl'), train_manifest=str(root / 'parent.jsonl'),
            style=str(root / 'style.json'), feature_cache=str(root / 'features'))
        (bundle / f'{name}.yaml').write_text(yaml.safe_dump(cfg))
    source = root / 'synthetic-source.py'
    source.write_text('# Synthetic fixture only\n')
    pins = {str(source): sha256(source)}
    original = dict(fitting_count=8, source_file_sha256=pins,
        artifact_sha256={str(p): sha256(p) for p in bundle.rglob('*') if p.is_file()})
    save_json(bundle / 'bundle.json', original)
    protocol_path = root / 'configs/saturated-fitting200/protocol.json'
    save_json(protocol_path, protocol)
    identity = dict(files=pins, protocol=dict(synthetic=True))
    summary = dict(feasibility_rail_passed=True, supported_nonzero_finite_adapter_views=8,
                   controlled_prerequisite_clearance=True)
    save_json(proof / 'summary.json', summary)
    state = dict(status='complete', current_check=None, completed=8, expected=8, optimizer_updates=0,
        parameters_unchanged=True, identity=identity, identity_hash=object_hash(identity),
        summary_sha256=sha256(proof / 'summary.json'))
    save_json(proof / 'status.json', state)
    (proof / 'checks').mkdir()
    monkeypatch.setattr(preparation, 'ROOT', root)
    monkeypatch.setattr(preparation, 'PROTOCOL', protocol_path)
    monkeypatch.setattr(preparation, 'ORIGINAL_BUNDLE', bundle)
    monkeypatch.setattr(preparation, 'OUTPUT', trial)
    monkeypatch.setattr(preparation, 'source_pins', lambda: pins)
    monkeypatch.setattr(preparation, 'provenance', lambda: dict(pipeline_source_hash='SYNTHETIC'))
    monkeypatch.setattr(preparation, 'verify_bundle', lambda directory: original)
    monkeypatch.setattr(preparation.feasibility, 'protocol_identity', lambda path: ({}, bundle, {}, rows, None, None, identity))
    monkeypatch.setattr(preparation.feasibility, 'verified_records', lambda output, state: records)
    monkeypatch.setattr(preparation.feasibility, 'aggregate', lambda records, state: summary)
    return dict(root=root, protocol=protocol, bundle=bundle, trial=trial, proof=proof,
                summary=summary, records=records, source=source, originals=originals, rows=rows)


def test_prepare_references_exact_original_labels_and_only_writes_configs(trial_inputs):
    fixture = trial_inputs
    record = preparation.prepare(fixture['trial'])
    assert preparation.verify_trial(fixture['trial']) == record
    assert {p.name for p in fixture['trial'].iterdir()} == {'trial.json', 'A0.yaml', 'A2.yaml', 'A3.yaml'}
    assert all(path.read_bytes() == before for path, before in fixture['originals'])
    configs = preparation.make_configs(fixture['trial'], fixture['protocol'], fixture['bundle'])
    assert configs['A2']['surface_weight'] == 0 and configs['A3']['surface_weight'] == .5
    assert configs['A2']['boundary_weight'] == configs['A3']['boundary_weight'] == 0
    assert all(cfg['fitting_bundle'] == str(fixture['bundle']) for cfg in configs.values())
    assert {k: v for k, v in configs['A2'].items() if k not in ('experiment', 'surface_weight')} == {
        k: v for k, v in configs['A3'].items() if k not in ('experiment', 'surface_weight')}
    with pytest.raises(ResearchError, match='fresh'):
        preparation.prepare(fixture['trial'])


@pytest.mark.parametrize('failure', ['unfinished', 'rail', 'five_views', 'mask_provenance', 'source'])
def test_preparation_blocks_incomplete_or_incompatible_feasibility(trial_inputs, failure):
    fixture = trial_inputs
    if failure == 'unfinished':
        state = load_json(fixture['proof'] / 'status.json')
        state['status'] = 'startup_complete'
        save_json(fixture['proof'] / 'status.json', state)
    elif failure in ('rail', 'five_views'):
        fixture['summary'].update(feasibility_rail_passed=failure != 'rail',
            supported_nonzero_finite_adapter_views=5 if failure == 'five_views' else 8)
        save_json(fixture['proof'] / 'summary.json', fixture['summary'])
        state = load_json(fixture['proof'] / 'status.json')
        state['summary_sha256'] = sha256(fixture['proof'] / 'summary.json')
        save_json(fixture['proof'] / 'status.json', state)
    elif failure == 'mask_provenance':
        fixture['records'][0]['surface_provenance']['building_array_sha256'] = '0' * 64
    else:
        assert fixture['source'].resolve().is_relative_to(fixture['root'])
        fixture['source'].write_text('SYNTHETIC-TAMPER')
    with pytest.raises(ResearchError):
        preparation.prepare(fixture['trial'])
    assert not fixture['trial'].exists()
    assert all(path.read_bytes() == before for path, before in fixture['originals'])


def test_trial_rejects_authenticated_config_tuning_or_label_copies(trial_inputs):
    fixture = trial_inputs
    preparation.prepare(fixture['trial'])
    cfg_path = fixture['trial'] / 'A3.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg['surface_weight'] = .25
    cfg_path.write_text(yaml.safe_dump(cfg))
    record = load_json(fixture['trial'] / 'trial.json')
    record['artifact_sha256'][str(cfg_path)] = sha256(cfg_path)
    save_json(fixture['trial'] / 'trial.json', record)
    with pytest.raises(ResearchError, match='configurations'):
        preparation.verify_trial(fixture['trial'])
    cfg['surface_weight'] = .5
    cfg_path.write_text(yaml.safe_dump(cfg))
    record['artifact_sha256'][str(cfg_path)] = sha256(cfg_path)
    save_json(fixture['trial'] / 'trial.json', record)
    (fixture['trial'] / 'labels').mkdir()
    with pytest.raises(ResearchError, match='copied labels'):
        preparation.verify_trial(fixture['trial'])


def test_fixed_budget_and_policy_cannot_be_changed(trial_inputs):
    protocol = copy.deepcopy(trial_inputs['protocol'])
    protocol['steps'] = 201
    with pytest.raises(ResearchError, match='fixed saturated'):
        preparation.make_configs(trial_inputs['trial'], protocol, trial_inputs['bundle'])


class CPUReference:
    def __init__(self):
        self.calls = 0
        self.segmenter = self
        self.boundary_mode = 'legacy'
        self.region_weight, self.boundary_weight, self.perceptual_weight = .5, 0., .1
        self.perceptual = lambda a, b: (a - b).square().mean().reshape(1)

    def probabilities(self, image):
        self.calls += 1
        return image.softmax(1)


def test_matched_wrapper_uses_one_semantic_forward_and_preserves_common_math():
    target = torch.zeros(1, 3, 16, 16)
    parameter = torch.nn.Parameter(torch.linspace(.1, .9, 768).reshape_as(target))
    building = torch.zeros(1, 1, 16, 16)
    building[:, :, 4:12, 3:13] = 1
    valid = torch.ones_like(building)
    labels = dict(building=building, valid=valid)
    prepared = preparation.prepare_surface_labels(building[0, 0].numpy(), valid[0, 0].numpy())
    cfg = dict(region_weight=.5, surface_weight=0., perceptual_weight=.1)
    reference = CPUReference()
    base, base_values = Objective.__call__(reference, parameter, target, labels)
    reference.calls = 0
    a2, a2_values = trainer.SaturatedObjective(cfg, reference)(parameter, target, labels, prepared)
    assert reference.calls == 1 and torch.equal(a2, base)
    assert all(a2_values[k] == base_values[k] for k in ('rgb', 'lpips', 'region', 'total'))
    base_gradient = torch.autograd.grad(base, parameter, retain_graph=True)[0]
    a2_gradient = torch.autograd.grad(a2, parameter, retain_graph=True)[0]
    assert torch.allclose(base_gradient, a2_gradient, rtol=1e-6, atol=1e-9)
    a3, a3_values = trainer.SaturatedObjective(dict(cfg, surface_weight=.5), reference)(parameter, target, labels, prepared)
    assert reference.calls == 2 and a3_values['surface'] == a2_values['surface']
    assert float(a3.detach()) == pytest.approx(float(a2.detach()) + .5 * a2_values['surface'])
    assert a2_values['boundary'] == a3_values['boundary'] == 0
    assert a2_values['legacy_boundary_diagnostic'] == base_values['boundary']
    gradient = torch.autograd.grad(a3, parameter)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0 and parameter.grad is None


def cpu_checkpoint(tmp_path, with_tail=False):
    adapter = StructuralAdapter(64, 16)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.00003, weight_decay=.0001)
    optimizer.zero_grad()
    sum((p - .3).square().sum() for p in adapter.parameters()).backward()
    optimizer.step()  # One synthetic CPU optimizer update, unrelated to the research renderer.
    rows = [dict(sample_id=f'SYNTHETIC-{i}') for i in range(8)]
    sampler = random.Random(17)
    sampled = [rows[sampler.randrange(8)]['sample_id'] for _ in range(8)]
    numpy_rng = np.random.RandomState(17).get_state()
    rng = dict(python=sampler.getstate(), numpy=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
               torch=torch.get_rng_state(), cuda=[torch.tensor([1, 2, 3], dtype=torch.uint8)])
    cfg = dict(seed=17, accumulation=8, amp=False, learning_rate=.00003, weight_decay=.0001,
               steps=200, experiment='A2')
    identity = dict(config=cfg)
    save_json(tmp_path / 'run.json', dict(identity=identity, identity_hash=object_hash(identity)))
    row = dict(step=1, sampled_view_ids=sampled, total=1., elapsed_seconds=1., peak_vram_mib=0.)
    state = dict(identity_hash=object_hash(identity), config=cfg, adapter=adapter.state_dict(),
        optimizer=optimizer.state_dict(), scaler={}, rng=rng, trainable_parameters=4448, step=1,
        elapsed_seconds=1., peak_vram_mib=0.)
    torch.save(state, tmp_path / 'last.pt')
    shutil.copyfile(tmp_path / 'last.pt', tmp_path / 'startup.pt')
    write_jsonl(tmp_path / 'training.jsonl', [row, dict(row, step=2)] if with_tail else [row])
    save_json(tmp_path / 'startup-receipt.json', dict(step=1, identity_hash=state['identity_hash'],
        first_log_row_sha256=object_hash(row), checkpoint_sha256=sha256(tmp_path / 'last.pt'),
        startup_checkpoint_sha256=sha256(tmp_path / 'startup.pt')))
    return state, rows, adapter, optimizer, row


def test_startup_checkpoint_resume_preserves_moments_rng_and_exact_sample_prefix(tmp_path):
    state, rows, _, _, _ = cpu_checkpoint(tmp_path, with_tail=True)
    restored = trainer.prepare_resume(tmp_path, state['identity_hash'], 200, rows)
    assert restored['step'] == 1 and restored['rng']['python'] == state['rng']['python']
    assert all(float(v['step']) == 1 for v in restored['optimizer']['state'].values())
    assert [r['step'] for r in read_jsonl(tmp_path / 'training.jsonl')] == [1]
    assert len(list(tmp_path.glob('interrupted-steps-*.jsonl'))) == 1
    replacement = StructuralAdapter(64, 16)
    replacement.load_state_dict(restored['adapter'])
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=.00003, weight_decay=.0001)
    optimizer.load_state_dict(restored['optimizer'])
    for key in state['optimizer']['state']:
        assert torch.equal(optimizer.state_dict()['state'][key]['exp_avg'], state['optimizer']['state'][key]['exp_avg'])


@pytest.mark.parametrize('failure', ['identity', 'sample', 'python_rng', 'missing_cuda_rng', 'moment', 'receipt'])
def test_bad_recovery_is_rejected_before_log_tail_mutation(tmp_path, failure):
    state, rows, _, _, _ = cpu_checkpoint(tmp_path, with_tail=True)
    before = (tmp_path / 'training.jsonl').read_bytes()
    identity = state['identity_hash']
    if failure == 'identity': identity = 'wrong'
    elif failure == 'sample':
        logs = read_jsonl(tmp_path / 'training.jsonl')
        logs[0]['sampled_view_ids'][0] = 'SYNTHETIC-WRONG'
        write_jsonl(tmp_path / 'training.jsonl', logs)
        before = (tmp_path / 'training.jsonl').read_bytes()
    elif failure == 'python_rng': state['rng']['python'] = random.Random(999).getstate()
    elif failure == 'missing_cuda_rng': state['rng']['cuda'] = []
    elif failure == 'moment': next(iter(state['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
    else:
        receipt = load_json(tmp_path / 'startup-receipt.json')
        receipt['checkpoint_sha256'] = '0' * 64
        save_json(tmp_path / 'startup-receipt.json', receipt)
    if failure in ('python_rng', 'missing_cuda_rng', 'moment'):
        torch.save(state, tmp_path / 'last.pt')
    with pytest.raises((ResearchError, ValueError)):
        trainer.prepare_resume(tmp_path, identity, 200, rows)
    assert (tmp_path / 'training.jsonl').read_bytes() == before
    assert not list(tmp_path.glob('interrupted-steps-*.jsonl'))


def test_startup_saves_last_only_and_keeps_update_inside_budget(tmp_path, monkeypatch):
    state, rows, adapter, optimizer, row = cpu_checkpoint(tmp_path)
    monkeypatch.setattr(trainer, 'rng_state', lambda: state['rng'])
    monkeypatch.setattr(trainer, 'check_pins', lambda pins: None)
    monkeypatch.setattr(trainer, 'provenance', lambda: dict(pipeline_source_hash='SYNTHETIC'))
    backend = type('CPUBackend', (), dict(adapter=adapter))()
    scaler = type('CPUScaler', (), dict(state_dict=lambda self: {}))()
    identity = dict(source_file_sha256={'synthetic': 'SYNTHETIC'}, pipeline_source_hash='SYNTHETIC',
        parent_training_manifest_sha256='SYNTHETIC', fitting_manifest_sha256='SYNTHETIC',
        data_digest='SYNTHETIC', readiness=dict(synthetic=True))
    result = trainer.save_checkpoint(tmp_path, backend, optimizer, scaler, state['config'], identity,
                                     state['identity_hash'], row, periodic=False)
    assert result['step'] == 1 and (tmp_path / 'last.pt').exists() and not list(tmp_path.glob('step-*.pt'))


def test_budget_complete_recovery_needs_no_gpu_restore_or_updates(tmp_path, monkeypatch):
    output = tmp_path / 'runs/completed'
    output.mkdir(parents=True)
    state, rows, _, _, first = cpu_checkpoint(output)
    sampler = random.Random(17)
    logs = []
    for step in range(1, 201):
        logs.append(dict(first, step=step, sampled_view_ids=[rows[sampler.randrange(8)]['sample_id'] for _ in range(8)]))
    state['step'], state['rng']['python'] = 200, sampler.getstate()
    for value in state['optimizer']['state'].values():
        value['step'].fill_(200)  # Synthetic authenticated final metadata, not research updates.
    torch.save(state, output / 'last.pt')
    write_jsonl(output / 'training.jsonl', logs)
    identity = load_json(output / 'run.json')['identity']
    monkeypatch.setattr(trainer, 'ROOT', tmp_path)
    monkeypatch.setattr(trainer, 'fitting_identity', lambda config, data: (state['config'], rows, identity))
    monkeypatch.setattr(runner, 'require_runner', lambda: None)
    monkeypatch.setattr(trainer, 'backend_from', lambda cfg: pytest.fail('GPU backend created for completed checkpoint'))
    result = trainer.train('SYNTHETIC', 'SYNTHETIC', output, resume=True)
    proof = load_json(output / 'resume-verification.json')
    assert result['step'] == 200 and proof['remaining_updates'] == 0
    assert proof['completion_without_optimizer_updates'] and not proof['adapter_optimizer_scaler_rng_restored']
    assert proof['checkpoint_payload_verified'] and proof['optimizer_updates_performed'] == 0


def test_exact_state_readback_detects_tensor_and_rng_changes_without_cuda():
    original = dict(adapter={'p': torch.tensor([1., 2.])}, rng={'python': (1, 2), 'cuda': [torch.tensor([1, 2], dtype=torch.uint8)]})
    assert trainer.exact_payload_equal(copy.deepcopy(original), original)
    changed = copy.deepcopy(original)
    changed['adapter']['p'][1] = 3
    assert not trainer.exact_payload_equal(changed, original)
    changed = copy.deepcopy(original)
    changed['rng']['cuda'][0][0] = 9
    assert not trainer.exact_payload_equal(changed, original)


def test_serial_stages_point_to_original_manifest_and_final_step_only(trial_inputs):
    fixture = trial_inputs
    preparation.prepare(fixture['trial'])
    stages = runner.make_stages(fixture['trial'], fixture['root'] / 'runs/run', fixture['root'] / 'rgb')
    assert [name for name, _ in stages] == ['A0-fitting', 'A0-eval', 'A2-train', 'A2-fitting', 'A2-eval',
                                         'A3-train', 'A3-fitting', 'A3-eval']
    for name, command in stages:
        if not name.endswith('-train'):
            assert command[command.index('--manifest') + 1] == str(fixture['bundle'] / 'manifest.jsonl')
        if name.endswith('-eval'):
            assert command[command.index('--manual-index') + 1] == str(fixture['bundle'] / 'manual-index.json')
    assert sum('train_saturated_fitting.py' in ' '.join(command) for _, command in stages) == 2


def test_direct_trainer_cannot_own_a_gpu_job(monkeypatch):
    monkeypatch.delenv('SATGROUND_SATURATED_RUNNER_TOKEN', raising=False)
    with pytest.raises(ResearchError, match='serial runner'):
        runner.require_runner()


def test_fresh_runner_requires_startup_before_preparation_or_gpu(monkeypatch):
    monkeypatch.setattr(runner, 'verify_trial', lambda path: pytest.fail('Trial verification reached without startup mode'))
    with pytest.raises(ResearchError, match='requires --startup-one-update'):
        runner.run()
