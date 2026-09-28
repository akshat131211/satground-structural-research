"""Synthetic CPU artifacts exercise fail-closed completion checks without model fitting."""
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import audit_reviewed_fitting_outputs as checker
from prepare_reviewed_fitting import prepare
from satground.adapter import StructuralAdapter
from satground.common import ResearchError, load_json, object_hash, provenance, save_json, sha256, write_jsonl
from verify_training_review import mask_measurements


@pytest.fixture
def completed(tmp_path):
    """Never touches research inputs; all records and tensors are synthetic pytest fixtures."""
    data, private = tmp_path / 'RGB', tmp_path / 'review'
    submitted = private / 'submitted'
    data.mkdir()
    submitted.mkdir(parents=True)
    rows, entries, pins = [], [], {}
    for i, number in enumerate((3, 4, 5, 6, 7, 9, 11, 12)):
        sid = f'SYNTHETIC-{i}'
        satellite, panorama = f'sat-{i}.png', f'pano-{i}.png'
        Image.fromarray(np.full((128, 512, 3), 60 + i, np.uint8)).save(data / panorama)
        Image.fromarray(np.full((256, 256, 3), 60 + i, np.uint8)).save(data / satellite)
        row = dict(sample_id=sid, city='Chicago', split='train', satellite=satellite, panorama=panorama,
                   geo_group=f'SYNTHETIC-group-{i}', size=256, yaw=0, pitch=0, fov=90)
        building = np.zeros((256, 256), bool)
        building[40:170, 35:160] = True
        valid = np.ones_like(building)
        paths = {}
        for key, pixels in (('target_image', np.full((256, 256, 3), 60 + i, np.uint8)),
                            ('building_mask', building.astype(np.uint8) * 255),
                            ('valid_mask', valid.astype(np.uint8) * 255)):
            path = submitted / f'{sid}-{key}.png'
            Image.fromarray(pixels).save(path)
            paths[key], pins[str(path)] = str(path), sha256(path)
        entries.append(dict(number=number, sample_id=sid, version=f'SYNTHETIC-v{i}', paths=paths,
            completed_utc='2026-09-28T00:00:00+00:00',
            human_review=dict(pairing='supported', support='mostly_inside', mask_review='checked', reviewer='SYNTHETIC'),
            measurements=mask_measurements(building, valid, building, valid)))
        rows.append(row)
    snapshot = dict(submissions=entries, source_file_sha256=pins)
    snapshot_path, decision = submitted / 'submissions.json', private / 'disposition.json'
    save_json(snapshot_path, snapshot)
    save_json(decision, dict(snapshot_content_hash=object_hash(snapshot), source='user_message',
        raw_user_answer='SYNTHETIC TEST ONLY', recorded_utc='2026-09-28T00:00:00+00:00',
        dispositions={str(e['number']): dict(action='include', reason='synthetic fixture') for e in entries}))
    parent, dev, style = private / 'train.jsonl', private / 'dev.jsonl', private / 'style.json'
    write_jsonl(parent, rows)
    write_jsonl(dev, [dict(rows[0], sample_id='SYNTHETIC-development', satellite='dev-sat', panorama='dev-pano',
                          geo_group='dev-group', split='validation')])
    save_json(style, dict(source_split='train', synthetic=False, manifest_sha256=sha256(parent), histogram=[1 / 90] * 270))
    bundle, root = private / 'bundle', tmp_path / 'SYNTHETIC-run'
    record = prepare(snapshot_path, decision, bundle, data, parent=parent, development=dev, style=style, expected_count=8)
    root.mkdir()
    status = dict(status='complete', completed=checker.STAGES, current_stage=None, child_pid=None,
        identity=dict(bundle=str(bundle), bundle_sha256=sha256(bundle / 'bundle.json'),
                      files={**record['source_file_sha256'], **record['artifact_sha256']},
                      pipeline_source_hash=provenance()['pipeline_source_hash'], training_identity={}))
    manual = load_json(bundle / 'manual-index.json')
    numpy_rng = np.random.RandomState(17).get_state()
    rng = dict(python=random.Random(17).getstate(), numpy=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
               torch=torch.Generator(device='cpu').manual_seed(17).get_state(), cuda=[torch.zeros(16, dtype=torch.uint8)])
    adapter = StructuralAdapter().state_dict()
    for name in ('A0', 'A2', 'A3'):
        images, evaluation = root / f'{name}-fitting', root / f'{name}-eval'
        images.mkdir()
        save_json(images / 'generation.json', {'synthetic': True})
        write_jsonl(images / 'predictions.jsonl', [{'sample_id': r['sample_id']} for r in rows])
        for folder in ('masks', 'targets', 'probabilities'):
            (evaluation / folder).mkdir(parents=True)
        for row in rows:
            sid, entry = row['sample_id'], manual[row['sample_id']]
            shutil.copyfile(entry['target_image'], images / (sid + '.png'))
            shutil.copyfile(entry['target_image'], evaluation / 'targets' / (sid + '.png'))
            shutil.copyfile(entry['building_mask'], evaluation / 'masks' / (sid + '.png'))
            np.save(evaluation / 'probabilities' / (sid + '.npy'), np.full((256, 256), .5, np.float32))
        if name == 'A0':
            continue
        train = root / f'{name}-train'
        train.mkdir()
        config = yaml.safe_load((bundle / f'{name}.yaml').read_text())
        identity = dict(config=config)
        identity_hash = object_hash(identity)
        status['identity']['training_identity'][name] = identity_hash
        save_json(train / 'run.json', dict(identity=identity, identity_hash=identity_hash))
        write_jsonl(train / 'training.jsonl', [dict(step=i, total=1., rgb=.2, lpips=.5, region=.3, boundary=.01,
            gradient_norm=.1, peak_vram_mib=1.,
            elapsed_seconds=float(i), sampled_view_ids=[r['sample_id'] for r in rows]) for i in range(1, 201)])
        for step in (50, 100, 150, 200):
            moments = {i: dict(step=torch.tensor(float(step)), exp_avg=torch.zeros_like(v), exp_avg_sq=torch.ones_like(v))
                       for i, v in enumerate(adapter.values())}
            state = dict(step=step, identity_hash=identity_hash, config=config, adapter=adapter, rng=rng, scaler={},
                trainable_parameters=4448, optimizer=dict(state=moments, param_groups=[dict(params=list(moments),
                    lr=config['learning_rate'], weight_decay=config['weight_decay'], amsgrad=False)]))
            torch.save(state, train / f'step-{step:06d}.pt')
        shutil.copyfile(train / 'step-000200.pt', train / 'last.pt')
        save_json(train / 'summary.json', dict(status='budget_complete', step=200, checkpoint_sha256=sha256(train / 'last.pt')))
    save_json(root / 'status.json', status)
    return root


def test_complete_cpu_audit_is_aggregate_only_and_fresh(completed, monkeypatch):
    # Any unexpected CUDA probe must fail this CPU-only test.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA called'))
    output = completed / 'audit.json'
    result = checker.audit(completed, output)
    assert result['probability_arrays_verified'] == 24
    assert result['periodic_checkpoints_verified'] == 8 and result['trainable_parameters'] == 4448
    assert result['cuda_operations'] == result['optimizer_updates_performed'] == 0
    text = output.read_text()
    assert all(token not in text for token in ('SYNTHETIC-', 'sample_id', 'geo_group', 'raw_user_answer', str(completed)))
    with pytest.raises(ResearchError, match='fresh'):
        checker.audit(completed, output)


@pytest.mark.parametrize('problem', ['status', 'stages', 'child', 'source_pin', 'missing_rgb', 'extra_mask',
    'rgb_size', 'mask_values', 'target_bytes', 'probability_dtype', 'probability_shape', 'probability_nan', 'probability_range'])
def test_incomplete_or_changed_output_is_rejected(completed, problem):
    if problem in ('status', 'stages', 'child'):
        path = completed / 'status.json'
        value = load_json(path)
        if problem == 'status':
            value['status'] = 'running'
        elif problem == 'stages':
            value['completed'].pop()
        else:
            value['child_pid'] = 123
        save_json(path, value)
    elif problem == 'source_pin':
        value = load_json(completed / 'status.json')
        manual = load_json(Path(value['identity']['bundle']) / 'manual-index.json')
        pinned = Path(manual['SYNTHETIC-0']['target_image']).resolve()
        assert pinned.is_relative_to(completed.parent.resolve())
        assert str(pinned) in value['identity']['files']
        pinned.write_bytes(pinned.read_bytes() + b'SYNTHETIC TAMPER')
    elif problem == 'missing_rgb':
        next((completed / 'A3-fitting').glob('*.png')).unlink()
    elif problem == 'extra_mask':
        Image.new('L', (256, 256)).save(completed / 'A3-eval/masks/EXTRA.png')
    elif problem == 'rgb_size':
        Image.new('RGB', (128, 256)).save(next((completed / 'A3-fitting').glob('*.png')))
    elif problem == 'mask_values':
        Image.fromarray(np.full((256, 256), 42, np.uint8)).save(next((completed / 'A3-eval/masks').glob('*.png')))
    elif problem == 'target_bytes':
        Image.new('RGB', (256, 256), (99, 99, 99)).save(next((completed / 'A3-eval/targets').glob('*.png')))
    else:
        path = next((completed / 'A3-eval/probabilities').glob('*.npy'))
        values = np.full((256, 256), .5, np.float32)
        if problem == 'probability_dtype':
            values = values.astype(np.float64)
        elif problem == 'probability_shape':
            values = values[:128]
        elif problem == 'probability_nan':
            values[1, 2] = np.nan
        else:
            values[1, 2] = 1.1
        np.save(path, values)
    with pytest.raises((ResearchError, RuntimeError, ValueError)):
        checker.audit(completed, completed / 'rejected.json')
    assert not (completed / 'rejected.json').exists()


@pytest.mark.parametrize('problem', ['adapter_keys', 'adapter_shape', 'parameter_count', 'optimizer_coverage',
    'optimizer_shape', 'optimizer_step', 'rng_missing', 'rng_invalid', 'scaler', 'nonfinite', 'last_mismatch', 'schedule',
    'loss_fields'])
def test_checkpoint_or_sampling_discrepancy_is_rejected(completed, problem):
    if problem in ('schedule', 'loss_fields'):
        from satground.common import read_jsonl
        path = completed / 'A3-train/training.jsonl'
        rows = read_jsonl(path)
        if problem == 'schedule':
            rows[0]['sampled_view_ids'].reverse()
        else:
            del rows[0]['rgb']
        write_jsonl(path, rows)
    elif problem == 'last_mismatch':
        path = completed / 'A3-train/last.pt'
        path.write_bytes(path.read_bytes() + b'SYNTHETIC TAMPER')
    else:
        path = completed / 'A3-train/step-000050.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        key = next(iter(state['adapter']))
        opt = state['optimizer']['state']
        if problem == 'adapter_keys':
            del state['adapter'][key]
        elif problem == 'adapter_shape':
            state['adapter'][key] = torch.zeros(1)
        elif problem == 'parameter_count':
            state['trainable_parameters'] = 4000
        elif problem == 'optimizer_coverage':
            del opt[0]
        elif problem == 'optimizer_shape':
            opt[0]['exp_avg'] = torch.zeros(1)
        elif problem == 'optimizer_step':
            opt[0]['step'] = torch.tensor(49.)
        elif problem == 'rng_missing':
            del state['rng']['cuda']
        elif problem == 'rng_invalid':
            state['rng']['torch'] = torch.zeros(1)
        elif problem == 'scaler':
            state['scaler'] = {'scale': 1024.}
        else:
            opt[0]['exp_avg'].flatten()[0] = float('inf')
        torch.save(state, path)
    with pytest.raises((ResearchError, RuntimeError, ValueError)):
        checker.audit(completed, completed / 'rejected.json')
    assert not (completed / 'rejected.json').exists()


def test_live_lock_and_runner_are_rejected(tmp_path, monkeypatch):
    lock = tmp_path / 'lock.json'
    lock.write_text('{}')
    monkeypatch.setattr(checker, 'LOCK', lock)
    with pytest.raises(ResearchError, match='lock'):
        checker.check_live_exit()
    lock.unlink()
    fake = SimpleNamespace(pid=999999, info={'name': 'python', 'cmdline': ['python', 'scripts/run_reviewed_fitting.py']},
                           cwd=lambda: str(checker.ROOT))
    monkeypatch.setattr(checker.psutil, 'process_iter', lambda fields: [fake])
    with pytest.raises(ResearchError, match='remains live'):
        checker.check_live_exit()
    monkeypatch.setattr(checker.psutil, 'process_iter', lambda fields: [])
    checker.check_live_exit()
