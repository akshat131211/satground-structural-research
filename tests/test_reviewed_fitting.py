"""Synthetic protocol tests; no research data, model download or GPU updates."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from prepare_reviewed_fitting import (PROTOCOL, make_configs, prepare, selected_reviews, verify_bundle)
from report_reviewed_fitting import aggregate, fitting_decision, report
from run_reviewed_fitting import make_stages, require_runner
from satground.camera import crop_panorama
from satground.common import ResearchError, load_json, object_hash, provenance, read_jsonl, save_json, sha256, write_jsonl
from satground.metrics import building_metrics, grouped_mean
from train_reviewed_fitting import finite_tree, prepare_resume
from verify_training_review import mask_measurements


def selection_fixture():
    entries = [dict(number=n, version=f'synthetic-v{n}', measurements={'interior_valid_boundary_pixels': 40},
                    human_review={'pairing': 'supported', 'support': 'mostly_inside', 'mask_review': 'checked'})
               for n in (1, 2, 3, 4, 8, 10)]
    entries[1]['human_review']['pairing'] = 'mismatch'
    entries[4]['human_review']['support'] = 'uncertain'
    entries[5]['measurements']['interior_valid_boundary_pixels'] = 0
    snapshot = {'submissions': entries}
    decision = dict(snapshot_content_hash=object_hash(snapshot), source='user_message',
        raw_user_answer='SYNTHETIC TEST: exclude unresolved views.', recorded_utc='2026-09-28T00:00:00+00:00',
        dispositions={str(e['number']): dict(action='include' if e['number'] in (3, 4) else 'exclude',
                                            reason='synthetic fixture') for e in entries})
    return snapshot, decision


def test_only_explicit_resolved_subset_is_selected():
    snapshot, decision = selection_fixture()
    assert [e['number'] for e in selected_reviews(snapshot, decision)] == [3, 4]
    assert [e['number'] for e in snapshot['submissions']] == [1, 2, 3, 4, 8, 10]


@pytest.mark.parametrize('number,match', [(1, 'View 1'), (2, 'mismatched'), (8, 'Uncertain'), (10, 'no usable')])
def test_known_unresolved_views_cannot_be_silently_included(number, match):
    snapshot, decision = selection_fixture()
    decision['dispositions'][str(number)]['action'] = 'include'
    with pytest.raises(ResearchError, match=match):
        selected_reviews(snapshot, decision)


@pytest.mark.parametrize('change', ['no_answer', 'wrong_snapshot', 'partial_disposition', 'pending'])
def test_resume_or_missing_answer_is_not_a_disposition(change):
    snapshot, decision = selection_fixture()
    if change == 'no_answer':
        decision['raw_user_answer'] = ''
    elif change == 'wrong_snapshot':
        decision['snapshot_content_hash'] = 'wrong'
    elif change == 'partial_disposition':
        del decision['dispositions']['4']
    else:
        decision['dispositions']['1']['action'] = 'pending'
    with pytest.raises(ResearchError):
        selected_reviews(snapshot, decision)


def fitting_files(tmp_path):
    data = tmp_path / 'rgb'
    private = tmp_path / 'review'
    submitted = private / 'submitted'
    data.mkdir()
    submitted.mkdir(parents=True)
    rows, entries, pins = [], [], {}
    for i in (3, 4):
        sid = f'SYNTHETIC-{i}'
        pano = np.full((128, 512, 3), 60 + i, np.uint8)
        satellite, panorama = f'satellite-{i}.png', f'panorama-{i}.png'
        Image.fromarray(pano).save(data / panorama)
        Image.fromarray(pano[:, :128]).save(data / satellite)
        row = dict(sample_id=sid, city='Chicago', split='train', satellite=satellite, panorama=panorama,
                   geo_group=f'SYNTHETIC-group-{i}', size=256, yaw=0, pitch=0, fov=90)
        target = crop_panorama(Image.fromarray(pano), 0, 0, 90, 256)
        building = np.zeros((256, 256), bool)
        building[30:170, 40:180] = True
        valid = np.ones_like(building)
        paths = {}
        for name, pixels in [('target_image', target), ('building_mask', building.astype(np.uint8) * 255),
                             ('valid_mask', valid.astype(np.uint8) * 255)]:
            path = submitted / f'{sid}-{name}.png'
            Image.fromarray(pixels).save(path)
            paths[name] = str(path)
            pins[str(path)] = sha256(path)
        entries.append(dict(number=i, sample_id=sid, version=f'SYNTHETIC-v{i}', paths=paths,
            completed_utc='2026-09-28T00:00:00+00:00',
            human_review=dict(pairing='supported', support='mostly_inside', mask_review='checked', reviewer='SYNTHETIC'),
            measurements=mask_measurements(building, valid, building, valid)))
        rows.append(row)
    snapshot = dict(submissions=entries, source_file_sha256=pins, human_submitted_masks_modified=False)
    snap = submitted / 'submissions.json'
    save_json(snap, snapshot)
    decision = private / 'decision.json'
    save_json(decision, dict(snapshot_content_hash=object_hash(snapshot), source='user_message',
        raw_user_answer='SYNTHETIC TEST ONLY', recorded_utc='2026-09-28T00:00:00+00:00',
        dispositions={str(e['number']): dict(action='include', reason='synthetic test') for e in entries}))
    parent, dev = private / 'train.jsonl', private / 'dev.jsonl'
    write_jsonl(parent, rows)
    write_jsonl(dev, [dict(rows[0], sample_id='SYNTHETIC-development', satellite='dev-sat', panorama='dev-pano',
                           geo_group='dev-group', split='validation')])
    style = private / 'style.json'
    save_json(style, dict(source_split='train', synthetic=False, manifest_sha256=sha256(parent), histogram=[1 / 90] * 270))
    return dict(snapshot_path=snap, decision_path=decision, output=private / 'bundle', data_root=data,
                parent=parent, development=dev, style=style, expected_count=2)


def test_bundle_preserves_exact_masks_and_distinguishes_style_source(tmp_path):
    kwargs = fitting_files(tmp_path)
    snapshot_bytes = kwargs['snapshot_path'].read_bytes()
    record = prepare(**kwargs)
    assert record['kind'] == 'human_reviewed_training_masks'
    assert record['fitting_count'] == 2 and not record['main_study_eligible']
    assert kwargs['snapshot_path'].read_bytes() == snapshot_bytes
    assert verify_bundle(kwargs['output']) == record
    import yaml
    cfg = yaml.safe_load((kwargs['output'] / 'A3.yaml').read_text())
    assert cfg['train_manifest'] == str(kwargs['parent'])
    assert cfg['fitting_manifest'] == str(kwargs['output'] / 'manifest.jsonl')
    assert cfg['train_manifest'] != cfg['fitting_manifest']
    with pytest.raises(ResearchError, match='fresh'):
        prepare(**kwargs)


def test_changed_reviewed_png_is_rejected_before_conversion(tmp_path):
    kwargs = fitting_files(tmp_path)
    path = Path(load_json(kwargs['snapshot_path'])['submissions'][0]['paths']['building_mask'])
    path.write_bytes(path.read_bytes() + b'SYNTHETIC TAMPER')
    with pytest.raises(ResearchError, match='changed'):
        prepare(**kwargs)
    assert not kwargs['output'].exists()


def test_converted_npz_cannot_diverge_even_if_its_hash_is_updated(tmp_path):
    kwargs = fitting_files(tmp_path)
    record = prepare(**kwargs)
    path = next((kwargs['output'] / 'labels').glob('*.npz'))
    with np.load(path) as values:
        building, valid = values['building'].copy(), values['valid'].copy()
    building[60, 70] = 0
    np.savez_compressed(path, building=building, valid=valid)
    record['artifact_sha256'][str(path)] = sha256(path)
    save_json(kwargs['output'] / 'bundle.json', record)
    with pytest.raises(ResearchError, match='differs from the submitted'):
        verify_bundle(kwargs['output'])


@pytest.mark.parametrize('problem', ['development_overlap', 'sealed_city', 'sealed_split', 'changed_crop'])
def test_wrong_fitting_cohort_is_rejected(tmp_path, problem):
    kwargs = fitting_files(tmp_path)
    if problem == 'development_overlap':
        from satground.common import read_jsonl
        row = read_jsonl(kwargs['development'])[0]
        row['satellite'] = 'satellite-3.png'
        write_jsonl(kwargs['development'], [row])
    elif problem in ('sealed_city', 'sealed_split'):
        from satground.common import read_jsonl
        rows = read_jsonl(kwargs['parent'])
        rows[0]['city' if problem == 'sealed_city' else 'split'] = 'Seattle' if problem == 'sealed_city' else 'audit'
        write_jsonl(kwargs['parent'], rows)
    else:
        Image.fromarray(np.full((128, 512, 3), 140, np.uint8)).save(kwargs['data_root'] / 'panorama-3.png')
    with pytest.raises(ResearchError):
        prepare(**kwargs)
    assert not kwargs['output'].exists()


def test_matched_configurations_cannot_extend_budget(tmp_path):
    protocol = load_json(PROTOCOL)
    cfg = make_configs(tmp_path, protocol)
    assert cfg['A2']['boundary_weight'] == 0 and cfg['A3']['boundary_weight'] == .5
    assert {k: v for k, v in cfg['A2'].items() if k not in ('experiment', 'boundary_weight')} == {
        k: v for k, v in cfg['A3'].items() if k not in ('experiment', 'boundary_weight')}
    protocol['steps'] = 201
    with pytest.raises(ResearchError, match='fixed 200'):
        make_configs(tmp_path, protocol)


def test_serial_stage_manifest_never_uses_development(tmp_path):
    save_json(tmp_path / 'bundle.json', {'protocol': load_json(PROTOCOL)})
    stages = make_stages(tmp_path, tmp_path / 'run', tmp_path / 'rgb')
    assert [n for n, _ in stages] == ['A0-fitting', 'A0-eval', 'A2-train', 'A2-fitting', 'A2-eval',
                                      'A3-train', 'A3-fitting', 'A3-eval']
    assert all('validation' not in ' '.join(cmd) and 'audit' not in ' '.join(cmd) for _, cmd in stages)
    assert sum('train_reviewed_fitting.py' in ' '.join(cmd) for _, cmd in stages) == 2


def test_unowned_direct_trainer_is_blocked(monkeypatch):
    monkeypatch.delenv('SATGROUND_FITTING_RUNNER_TOKEN', raising=False)
    with pytest.raises(ResearchError, match='serial runner'):
        require_runner()


def checkpoint_fixture(tmp_path):
    # A real CPU optimizer state exercises deserialization/recovery, not model fitting.
    parameter = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([parameter], lr=.01)
    for _ in range(2):
        opt.zero_grad()
        (parameter - 1).square().sum().backward()
        opt.step()
    identity = {'config': {'synthetic': True}}
    save_json(tmp_path / 'run.json', {'identity': identity, 'identity_hash': object_hash(identity)})
    state = dict(identity_hash=object_hash(identity), config=identity['config'], step=2, adapter={'p': parameter.detach()},
                 optimizer=opt.state_dict(), scaler={}, rng={'synthetic': True})
    torch.save(state, tmp_path / 'last.pt')
    write_jsonl(tmp_path / 'training.jsonl', [dict(step=i, total=float(i)) for i in range(1, 4)])
    return state


def test_resume_preserves_interrupted_tail_and_restores_optimizer(tmp_path):
    state = checkpoint_fixture(tmp_path)
    restored = prepare_resume(tmp_path, state['identity_hash'], 200)
    assert restored['step'] == 2
    assert torch.equal(restored['adapter']['p'], state['adapter']['p'])
    assert float(next(iter(restored['optimizer']['state'].values()))['step']) == 2
    assert len(list(tmp_path.glob('interrupted-steps-*.jsonl'))) == 1
    from satground.common import read_jsonl
    assert [r['step'] for r in read_jsonl(tmp_path / 'training.jsonl')] == [1, 2]


def test_resume_rejects_identity_or_nonfinite_checkpoint_without_rewriting_logs(tmp_path):
    state = checkpoint_fixture(tmp_path)
    before = (tmp_path / 'training.jsonl').read_bytes()
    with pytest.raises(ResearchError, match='altered'):
        prepare_resume(tmp_path, 'WRONG', 200)
    assert (tmp_path / 'training.jsonl').read_bytes() == before
    state['adapter']['p'][0] = float('nan')
    torch.save(state, tmp_path / 'last.pt')
    with pytest.raises(ResearchError, match='nonfinite'):
        prepare_resume(tmp_path, state['identity_hash'], 200)
    assert (tmp_path / 'training.jsonl').read_bytes() == before


def metric_rows(boundary):
    return [dict(sample_id=f'SYNTHETIC-{i}', geo_group=f'SYNTHETIC-group-{i}', boundary_error=boundary,
                 building_iou=.6, lpips=.5, ssim=.4, target_building_pixels=25, predicted_building_pixels=26,
                 false_positive_fraction=.02) for i in (3, 4)]


def test_fitting_decision_requires_quality_and_both_references():
    protocol = load_json(PROTOCOL)
    rows = {'A0': metric_rows(.1), 'A2': metric_rows(.09), 'A3': metric_rows(.07)}
    decision = fitting_decision(rows, protocol['decision'])
    assert decision['status'] == 'fitting_response_only'
    assert not decision['server_gate_eligible'] and not decision['held_out_generalization_evidence']
    rows['A3'][0]['lpips'] = .8
    assert fitting_decision(rows, protocol['decision'])['status'] == 'fitting_response_not_established'
    rows['A3'] = metric_rows(.07)
    rows['A0'] = metric_rows(.01)
    assert fitting_decision(rows, protocol['decision'])['status'] == 'fitting_response_not_established'


def test_zero_reference_is_undefined_and_absent_stratum_is_not_zero_error():
    protocol = load_json(PROTOCOL)
    rows = {'A0': metric_rows(0), 'A2': metric_rows(.09), 'A3': metric_rows(.07)}
    result = fitting_decision(rows, protocol['decision'])
    assert result['comparisons']['A0']['relative_boundary_reduction'] is None
    assert result['status'] == 'fitting_response_not_established'
    assert aggregate(rows['A3'])['target_empty']['metrics'] is None
    assert not finite_tree({'nested': [torch.tensor(float('inf'))]})


def completed_run_fixture(tmp_path):
    """Fabricated records exclusively inside a pytest temporary fixture, never a research run."""
    import shutil
    import yaml
    kwargs = fitting_files(tmp_path)
    bundle_record = prepare(**kwargs)
    bundle = kwargs['output']
    root = tmp_path / 'SYNTHETIC-run'
    root.mkdir()
    manual = load_json(bundle / 'manual-index.json')
    manifest = read_jsonl(bundle / 'manifest.jsonl')
    ids = [r['sample_id'] for r in manifest]
    p = provenance()
    status = dict(status='complete', completed=[n for n, _ in make_stages(bundle, root, kwargs['data_root'])],
        identity=dict(bundle=str(bundle), bundle_sha256=sha256(bundle / 'bundle.json'),
                      files=bundle_record['artifact_sha256'], pipeline_source_hash=p['pipeline_source_hash'], training_identity={}))
    for name in ('A0', 'A2', 'A3'):
        cfg = yaml.safe_load((bundle / f'{name}.yaml').read_text())
        if name != 'A0':
            train = root / f'{name}-train'
            train.mkdir()
            identity = {'config': cfg}
            identity_hash = object_hash(identity)
            status['identity']['training_identity'][name] = identity_hash
            save_json(train / 'run.json', dict(identity=identity, identity_hash=identity_hash))
            write_jsonl(train / 'training.jsonl', [dict(step=step, total=1., gradient_norm=.1,
                peak_vram_mib=1., elapsed_seconds=float(step), sampled_view_ids=ids * 4) for step in range(1, 201)])
            for step in (50, 100, 150, 200):
                state = dict(config=cfg, identity_hash=identity_hash, step=step, adapter={'synthetic': torch.zeros(2)},
                             optimizer={'state': {0: {'step': torch.tensor(float(step))}}})
                torch.save(state, train / f'step-{step:06d}.pt')
            shutil.copyfile(train / 'step-000200.pt', train / 'last.pt')
            save_json(train / 'summary.json', dict(status='budget_complete', step=200, checkpoint_sha256=sha256(train / 'last.pt')))
        images, evaluation = root / f'{name}-fitting', root / f'{name}-eval'
        images.mkdir()
        (evaluation / 'masks').mkdir(parents=True)
        generation = dict(config=cfg, manifest_sha256=bundle_record['fitting_manifest_sha256'], scientific=True,
            target_images_used_for_conditioning=False, stress='clean', provenance=p,
            style_sha256=sha256(kwargs['style']), train_manifest_sha256=bundle_record['parent_training_manifest_sha256'],
            trained_steps=0 if name == 'A0' else 200,
            checkpoint_sha256=None if name == 'A0' else sha256(root / f'{name}-train/last.pt'),
            training_identity_hash=None if name == 'A0' else status['identity']['training_identity'][name])
        save_json(images / 'generation.json', generation)
        predictions, metrics = [], []
        for row in manifest:
            sid = row['sample_id']
            entry = manual[sid]
            shutil.copyfile(entry['target_image'], images / f'{sid}.png')
            shutil.copyfile(entry['building_mask'], evaluation / 'masks' / f'{sid}.png')
            predictions.append(dict(sample_id=sid, prediction=f'{sid}.png', sha256=sha256(images / f'{sid}.png')))
            with Image.open(entry['building_mask']) as image:
                building = np.asarray(image) == 255
            with Image.open(entry['valid_mask']) as image:
                valid = np.asarray(image) == 255
            metrics.append(dict(sample_id=sid, geo_group=row['geo_group'], split='train', label_provenance='manually_reviewed',
                **building_metrics(building, building, valid), lpips=.5, ssim=.4))
        write_jsonl(images / 'predictions.jsonl', predictions)
        write_jsonl(evaluation / 'metrics.jsonl', metrics)
        save_json(evaluation / 'summary.json', dict(manifest_sha256=bundle_record['fitting_manifest_sha256'],
            generation=generation, generation_sha256=sha256(images / 'generation.json'),
            manual_index_sha256=sha256(bundle / 'manual-index.json'), segmenter_revision=bundle_record['protocol']['evaluation_segmenter_revision'],
            training_segmenter_different=True, manually_reviewed_count=2, evaluation_provenance=p,
            group_weighted={k: grouped_mean(metrics, k) for k in ('boundary_error', 'building_iou', 'lpips', 'ssim')}))
    save_json(root / 'status.json', status)
    return root


def test_complete_report_reproduces_and_publishes_no_sample_ids(tmp_path):
    root = completed_run_fixture(tmp_path)
    first, second = report(root, root / 'report-1'), report(root, root / 'report-2')
    assert first == second
    assert first['training']['A2']['updates'] == first['training']['A3']['updates'] == 200
    assert first['decision']['status'] == 'fitting_response_not_established'
    assert first['manual_gallery_review_pending'] and not first['server_gate_eligible']
    text = (root / 'report-1/aggregate.json').read_text()
    assert all(token not in text for token in ('SYNTHETIC-3', 'SYNTHETIC-4', 'sample_id', 'geo_group', 'raw_user_answer'))
    assert (root / 'report-1/paired-gallery.png').is_file()


@pytest.mark.parametrize('problem', ['missing_update', 'unmatched_schedule', 'changed_prediction', 'wrong_aggregate', 'changed_mask'])
def test_report_rejects_material_evidence_discrepancies(tmp_path, problem):
    root = completed_run_fixture(tmp_path)
    if problem in ('missing_update', 'unmatched_schedule'):
        path = root / 'A3-train/training.jsonl'
        rows = read_jsonl(path)
        if problem == 'missing_update':
            rows.pop()
        else:
            rows[0]['sampled_view_ids'].reverse()
        write_jsonl(path, rows)
    elif problem == 'changed_prediction':
        path = next((root / 'A3-fitting').glob('*.png'))
        path.write_bytes(path.read_bytes() + b'SYNTHETIC TAMPER')
    elif problem == 'wrong_aggregate':
        path = root / 'A3-eval/summary.json'
        record = load_json(path)
        record['group_weighted']['lpips'] = .01
        save_json(path, record)
    else:
        path = next((root / 'A3-eval/masks').glob('*.png'))
        with Image.open(path) as image:
            pixels = np.array(image)
        pixels[80, 90] = 0
        Image.fromarray(pixels).save(path)
    with pytest.raises(ResearchError):
        report(root, root / 'rejected-report')
    assert not (root / 'rejected-report').exists()
