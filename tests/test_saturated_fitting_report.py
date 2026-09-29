"""Synthetic final-only saturated fitting report tests; no model or GPU run."""
from copy import deepcopy
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import report_saturated_fitting as reporter
from diagnose_fitting_boundaries import boundary_counts
from prepare_reviewed_fitting import prepare, verify_bundle
from prepare_saturated_fitting import make_configs
from satground.adapter import StructuralAdapter
from satground.camera import crop_panorama
from satground.common import (ResearchError, load_json, object_hash, provenance, read_jsonl,
                             save_json, sha256, write_jsonl)
from satground.metrics import building_metrics, grouped_mean
from verify_training_review import mask_measurements

PROTOCOL = Path(__file__).parents[1] / 'configs/saturated-fitting200/protocol.json'


def metric_rows(full, nonperimeter, *, iou=.6, lpips=.5):
    return [dict(sample_id=f'SYNTHETIC-{i}', geo_group=f'SYNTHETIC-group-{i}', boundary_error=full,
                 nonperimeter_boundary_error=nonperimeter, building_iou=iou, lpips=lpips, ssim=.4,
                 target_building_pixels=25, predicted_building_pixels=26, false_positive_fraction=.02,
                 primary_target_edge_pixels=20, target_perimeter_edge_pixels=4, target_nonperimeter_edge_pixels=16,
                 predicted_nonperimeter_edge_pixels=16, full_boundary_empty_case='neither_empty',
                 nonperimeter_empty_case='neither_empty') for i in range(8)]


def test_co_primary_decision_requires_both_metrics_both_references_quality_and_joint_views():
    thresholds = load_json(PROTOCOL)['decision']
    rows = dict(A0=metric_rows(.1, .12), A2=metric_rows(.09, .1), A3=metric_rows(.07, .07))
    result = reporter.fitting_decision(rows, thresholds)
    assert result['status'] == 'fitting_response_only'
    assert result['comparisons']['A2']['jointly_improved_views'] == 8
    assert result['comparisons']['A2']['geographic_intervals']['boundary_error']['geographic_groups'] == 8
    assert not result['server_gate_eligible'] and not result['held_out_generalization_evidence']
    cases = []
    a = deepcopy(rows); a['A3'] = metric_rows(.07, .095); cases.append(a)  # full-only gain.
    a = deepcopy(rows); a['A0'] = metric_rows(.03, .04); cases.append(a)  # control gain, frozen reference worsens.
    a = deepcopy(rows); a['A3'] = metric_rows(.07, .07, iou=.58); cases.append(a)
    a = deepcopy(rows); a['A3'] = metric_rows(.07, .07, lpips=.54); cases.append(a)
    for value in cases:
        assert reporter.fitting_decision(value, thresholds)['status'] == 'fitting_response_not_established'


def test_joint_fraction_cannot_be_replaced_by_separate_successful_halves():
    rows = dict(A0=metric_rows(.2, .2), A2=metric_rows(.2, .2), A3=metric_rows(.01, .01))
    # Different halves improve different metrics. Aggregate improvements exceed
    # thresholds, but none jointly improves both metric outcomes.
    for i, row in enumerate(rows['A3']):
        row['boundary_error' if i < 4 else 'nonperimeter_boundary_error'] = .21
    result = reporter.fitting_decision(rows, load_json(PROTOCOL)['decision'])
    assert result['checks']['A2_full_boundary'] and result['checks']['A2_nonperimeter_boundary']
    assert result['comparisons']['A2']['jointly_improved_views'] == 0
    assert not result['checks']['joint_fraction_views_improved']
    assert result['status'] == 'fitting_response_not_established'


def test_zero_baselines_remain_undefined_and_absent_empty_stratum_not_zero():
    rows = dict(A0=metric_rows(0, 0), A2=metric_rows(.09, .1), A3=metric_rows(.07, .07))
    result = reporter.fitting_decision(rows, load_json(PROTOCOL)['decision'])
    assert result['comparisons']['A0']['relative_boundary_reduction'] is None
    assert result['comparisons']['A0']['relative_nonperimeter_reduction'] is None
    assert result['status'] == 'fitting_response_not_established'
    aggregate = reporter.aggregate(rows['A3'])
    assert aggregate['target_empty']['views'] == 0
    assert aggregate['target_empty']['metrics'] is None
    assert aggregate['target_empty']['false_positive_fraction'] is None
    empty = deepcopy(rows['A3'][0])
    empty.update(target_building_pixels=0, primary_target_edge_pixels=0, target_perimeter_edge_pixels=0,
                 target_nonperimeter_edge_pixels=0, full_boundary_empty_case='one_empty', nonperimeter_empty_case='one_empty')
    actual = reporter.aggregate([empty])['target_empty']
    assert actual['views'] == 1 and actual['metrics'] is not None
    assert actual['full_missing_or_extra_boundary_penalty_views'] == 1


def _prepare_original_eight(tmp_path):
    """Create clearly synthetic data wholly inside pytest temporary storage."""
    data, private = tmp_path / 'SYNTHETIC-rgb', tmp_path / 'SYNTHETIC-review'
    data.mkdir(); submitted = private / 'submitted'; submitted.mkdir(parents=True)
    rows, entries, pins = [], [], {}
    for ordinal, number in enumerate((3, 4, 5, 6, 7, 9, 11, 12)):
        sid = f'SYNTHETIC-view-{number}'
        pano = np.full((128, 512, 3), 60 + ordinal, np.uint8)
        sat, pano_name = f'SYNTHETIC-sat-{number}.png', f'SYNTHETIC-pano-{number}.png'
        Image.fromarray(pano).save(data / pano_name); Image.fromarray(pano[:, :128]).save(data / sat)
        row = dict(sample_id=sid, city='Chicago', split='train', satellite=sat, panorama=pano_name,
                   geo_group=f'SYNTHETIC-location-{number}', size=256, yaw=0, pitch=0, fov=90)
        rows.append(row)
        target = crop_panorama(Image.fromarray(pano), 0, 0, 90, 256)
        building, valid = np.zeros((256, 256), bool), np.ones((256, 256), bool)
        building[40:170, 50:180] = True
        paths = {}
        for name, pixels in [('target_image', target), ('building_mask', building.astype(np.uint8) * 255),
                             ('valid_mask', valid.astype(np.uint8) * 255)]:
            path = submitted / f'{sid}-{name}.png'; Image.fromarray(pixels).save(path)
            paths[name], pins[str(path)] = str(path), sha256(path)
        entries.append(dict(number=number, sample_id=sid, version='SYNTHETIC-reviewed-v1', paths=paths,
            completed_utc='2026-09-29T00:00:00+00:00',
            human_review=dict(pairing='supported', support='mostly_inside', mask_review='checked', reviewer='SYNTHETIC ONLY'),
            measurements=mask_measurements(building, valid, building, valid)))
    snapshot = dict(submissions=entries, source_file_sha256=pins, human_submitted_masks_modified=False)
    snapshot_path, decision_path = submitted / 'submissions.json', private / 'decision.json'
    save_json(snapshot_path, snapshot)
    save_json(decision_path, dict(snapshot_content_hash=object_hash(snapshot), source='user_message',
        raw_user_answer='SYNTHETIC TEST FIXTURE ONLY', recorded_utc='2026-09-29T00:00:00+00:00',
        dispositions={str(entry['number']): dict(action='include', reason='synthetic unit fixture') for entry in entries}))
    parent, development, style = private / 'train.jsonl', private / 'development.jsonl', private / 'style.json'
    write_jsonl(parent, rows)
    write_jsonl(development, [dict(rows[0], sample_id='SYNTHETIC-unused-dev', satellite='SYNTHETIC-dev-sat',
        panorama='SYNTHETIC-dev-pano', geo_group='SYNTHETIC-dev-location', split='validation')])
    save_json(style, dict(source_split='train', synthetic=False, manifest_sha256=sha256(parent), histogram=[1 / 90] * 270))
    original = private / 'original-eight'
    record = prepare(snapshot_path, decision_path, original, data, parent=parent, development=development, style=style, expected_count=8)
    return original, record, rows


def completed_fixture(tmp_path, monkeypatch):
    import yaml
    original, original_record, manifest = _prepare_original_eight(tmp_path)
    protocol = load_json(PROTOCOL)
    trial = tmp_path / 'SYNTHETIC-trial'; trial.mkdir()
    configs = make_configs(trial, protocol, original)
    for name, cfg in configs.items():
        (trial / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True), encoding='utf-8')
    reporter_path = Path(reporter.__file__)
    sources = {**original_record['source_file_sha256'], **original_record['artifact_sha256'],
               str(original / 'bundle.json'): sha256(original / 'bundle.json'), str(reporter_path): sha256(reporter_path)}
    record = dict(protocol=protocol, original_bundle=str(original), original_bundle_sha256=sha256(original / 'bundle.json'),
        source_file_sha256=sources, artifact_sha256={str(trial / f'{name}.yaml'): sha256(trial / f'{name}.yaml') for name in configs},
        pipeline_source_hash=provenance()['pipeline_source_hash'])
    save_json(trial / 'trial.json', record)

    def synthetic_trial_verification(path):
        assert Path(path) == trial
        value = load_json(trial / 'trial.json')
        reporter.check_pins(value['source_file_sha256']); reporter.check_pins(value['artifact_sha256'])
        assert sha256(original / 'bundle.json') == value['original_bundle_sha256']
        assert verify_bundle(original)['fitting_count'] == 8
        return value
    monkeypatch.setattr(reporter, 'verify_trial', synthetic_trial_verification)
    root = tmp_path / 'SYNTHETIC-run'; root.mkdir()
    p = provenance()
    manual = load_json(original / 'manual-index.json')
    runner = dict(trial=str(trial), bundle=str(original), bundle_sha256=record['original_bundle_sha256'],
        files={**sources, **record['artifact_sha256'], str(trial / 'trial.json'): sha256(trial / 'trial.json')},
        training_identity={}, pipeline_source_hash=p['pipeline_source_hash'])
    sampler = random.Random(17); schedule, sample_states = [], {}
    for step in range(1, 201):
        schedule.append([manifest[sampler.randrange(8)]['sample_id'] for _ in range(8)])
        sample_states[step] = sampler.getstate()
    with torch.random.fork_rng(devices=[]):
        adapter = StructuralAdapter(64, 16).state_dict()
    numpy_rng = np.random.RandomState(17).get_state()
    numpy_rng = [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]]
    startup, proof = None, None
    for name, config in configs.items():
        if name != 'A0':
            folder = root / f'{name}-train'; folder.mkdir()
            identity = dict(config=config, trial_sha256=sha256(trial / 'trial.json'), bundle_sha256=record['original_bundle_sha256'],
                bundle_inputs=sources, bundle_artifacts=record['artifact_sha256'],
                source_file_sha256={str(reporter_path): sha256(reporter_path)}, pipeline_source_hash=p['pipeline_source_hash'],
                fitting_manifest_sha256=sha256(original / 'manifest.jsonl'), parent_training_manifest_sha256=sha256(config['train_manifest']),
                style_sha256=sha256(config['style']), data_digest='SYNTHETIC-DATA-DIGEST')
            ih = object_hash(identity); runner['training_identity'][name] = ih
            save_json(folder / 'run.json', dict(identity=identity, identity_hash=ih))
            log = [dict(step=step, sampled_view_ids=schedule[step - 1], rgb=.2, lpips=.5, region=.4,
                boundary=0., boundary_weight=0., surface=.2, surface_weight=config['surface_weight'],
                legacy_boundary_diagnostic=.01, total=.2 + .1 * .5 + .5 * .4 + config['surface_weight'] * .2,
                gradient_norm=.1, peak_vram_mib=1., elapsed_seconds=float(step)) for step in range(1, 201)]
            write_jsonl(folder / 'training.jsonl', log)
            for step in (50, 100, 150, 200):
                optimizer = dict(state={i: dict(step=torch.tensor(float(step)), exp_avg=torch.zeros_like(value),
                    exp_avg_sq=torch.zeros_like(value)) for i, value in enumerate(adapter.values())},
                    param_groups=[dict(params=list(range(len(adapter))), lr=.00003, weight_decay=.0001, amsgrad=False)])
                state = dict(step=step, config=config, identity_hash=ih, adapter=adapter, optimizer=optimizer,
                    scaler={}, trainable_parameters=4448, rng=dict(python=sample_states[step], numpy=numpy_rng,
                    torch=torch.get_rng_state(), cuda=[torch.arange(64, dtype=torch.uint8)]),
                    fitting_manifest_sha256=identity['fitting_manifest_sha256'], train_manifest_sha256=identity['parent_training_manifest_sha256'],
                    train_data_digest=identity['data_digest'], provenance=p)
                torch.save(state, folder / f'step-{step:06d}.pt')
            shutil.copyfile(folder / 'step-000200.pt', folder / 'last.pt')
            save_json(folder / 'summary.json', dict(status='budget_complete', step=200, checkpoint_sha256=sha256(folder / 'last.pt')))
            if name == 'A2':
                # This explicitly synthetic CPU payload models the retained
                # first update. No optimizer or CUDA operation is executed.
                first = deepcopy(state)
                first['step'], first['rng']['python'] = 1, sample_states[1]
                for values in first['optimizer']['state'].values(): values['step'] = torch.tensor(1.)
                torch.save(first, folder / 'startup.pt')
                first_sha = sha256(folder / 'startup.pt')
                startup = dict(status='startup_complete', step=1, identity_hash=ih, budget_steps=200, remaining_updates=199,
                    updates_included_in_budget=True, sample_prefix_verified=True,
                    first_log_row_sha256=object_hash(log[0]), checkpoint_sha256=first_sha, startup_checkpoint_sha256=first_sha)
                proof = dict(identity_hash=ih, restored_step=1, remaining_updates=199, sample_prefix_verified=True,
                    checkpoint_payload_verified=True, adapter_optimizer_scaler_rng_restored=True,
                    exact_adapter_optimizer_scaler_readback_verified=True,
                    exact_python_numpy_torch_cpu_cuda_rng_readback_verified=True,
                    completion_without_optimizer_updates=False, optimizer_updates_performed=0,
                    checkpoint_sha256=first_sha)
                save_json(folder / 'startup-receipt.json', startup); save_json(folder / 'resume-verification.json', proof)
        images, evaluation = root / f'{name}-fitting', root / f'{name}-eval'
        images.mkdir()
        for subdir in ('masks', 'targets', 'probabilities'): (evaluation / subdir).mkdir(parents=True)
        generation = dict(config=config, manifest_sha256=original_record['fitting_manifest_sha256'], scientific=True,
            target_images_used_for_conditioning=False, stress='clean', provenance=p,
            style_sha256=sha256(config['style']), train_manifest_sha256=original_record['parent_training_manifest_sha256'],
            trained_steps=0 if name == 'A0' else 200, checkpoint_sha256=None if name == 'A0' else sha256(root / f'{name}-train/last.pt'),
            training_identity_hash=None if name == 'A0' else runner['training_identity'][name])
        save_json(images / 'generation.json', generation)
        predictions, metrics = [], []
        for row in manifest:
            sid, entry = row['sample_id'], manual[row['sample_id']]
            shutil.copyfile(entry['target_image'], images / f'{sid}.png')
            shutil.copyfile(entry['target_image'], evaluation / 'targets' / f'{sid}.png')
            target = np.asarray(Image.open(entry['building_mask'])) == 255
            valid = np.asarray(Image.open(entry['valid_mask'])) == 255
            mask = np.roll(target, {'A0': 8, 'A2': 6, 'A3': 2}[name], axis=1)
            Image.fromarray(mask.astype(np.uint8) * 255).save(evaluation / 'masks' / f'{sid}.png')
            np.save(evaluation / 'probabilities' / f'{sid}.npy', np.where(mask, .9, .1).astype(np.float32))
            predictions.append(dict(sample_id=sid, prediction=f'{sid}.png', sha256=sha256(images / f'{sid}.png')))
            metrics.append(dict(sample_id=sid, geo_group=row['geo_group'], split='train', label_provenance='manually_reviewed',
                **building_metrics(mask, target, valid), lpips=.5, ssim=.4))
        write_jsonl(images / 'predictions.jsonl', predictions); write_jsonl(evaluation / 'metrics.jsonl', metrics)
        save_json(evaluation / 'summary.json', dict(count=8, manifest_sha256=original_record['fitting_manifest_sha256'],
            generation=generation, generation_sha256=sha256(images / 'generation.json'), manual_index_sha256=sha256(original / 'manual-index.json'),
            segmenter='SYNTHETIC-INDEPENDENT-SEGMENTER', segmenter_revision=protocol['evaluation_segmenter_revision'],
            training_segmenter_different=True, manually_reviewed_count=8, evaluation_provenance=p,
            empty_mask_convention='both_empty: IoU1,error0; one_empty: error1',
            group_weighted={key: grouped_mean(metrics, key) for key in ('boundary_error', 'building_iou', 'lpips', 'ssim')}))
    status = dict(status='complete', current_stage=None, child_pid=None, identity=runner, completed=reporter.STAGES,
        startup_update=1, startup_checkpoint_sha256=startup['checkpoint_sha256'], verified_resumes={'A2-train': proof})
    save_json(root / 'status.json', status)
    return root


def set_latest_proof(root, proof):
    save_json(root / 'A2-train/resume-verification.json', proof)
    path = root / 'status.json'; status = load_json(path)
    status['verified_resumes']['A2-train'] = proof; save_json(path, status)


def repin_synthetic_startup(root):
    """Reach payload checks even if deliberately malformed fixture bytes repin."""
    digest = sha256(root / 'A2-train/startup.pt')
    path = root / 'A2-train/startup-receipt.json'; receipt = load_json(path)
    receipt['checkpoint_sha256'] = receipt['startup_checkpoint_sha256'] = digest; save_json(path, receipt)
    path = root / 'status.json'; status = load_json(path); status['startup_checkpoint_sha256'] = digest; save_json(path, status)
    proof = load_json(root / 'A2-train/resume-verification.json'); proof['checkpoint_sha256'] = digest; set_latest_proof(root, proof)


def test_full_report_verifies_payloads_reproduces_and_publishes_no_sample_details(tmp_path, monkeypatch):
    root = completed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('Synthetic reporter must never initialize CUDA.'))
    cpu_rng_before = torch.get_rng_state().clone()
    first = reporter.report(root, root / 'report-1')
    second = reporter.report(root, root / 'report-2')
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert first == second
    assert first['decision']['status'] == 'fitting_response_only'
    assert first['completion_audit']['training_rows_verified'] == 400
    assert first['completion_audit']['probability_arrays_verified'] == 24
    assert first['completion_audit']['periodic_checkpoints_verified'] == 8
    assert first['startup_resume']['retained_resume_proof_verified']
    assert first['startup_resume']['original_startup_checkpoint_bytes_reinspected']
    assert first['startup_resume']['actual_first_update_restore_proof_verified']
    assert first['startup_resume']['latest_actual_restore_performed']
    assert first['completion_audit']['original_startup_checkpoint_bytes_reinspected']
    assert first['models']['A3']['target_empty']['metrics'] is None
    assert first['manual_gallery_review_pending'] and not first['server_gate_eligible']
    text = (root / 'report-1/aggregate.json').read_text()
    assert all(value not in text for value in ('SYNTHETIC-view-', 'SYNTHETIC-location-', 'sample_id', 'geo_group', 'raw_user_answer', str(tmp_path)))
    with Image.open(root / 'report-1/paired-gallery.png') as image:
        assert image.size == (1024, 8 * 310)
    with pytest.raises(ResearchError, match='fresh'):
        reporter.report(root, root / 'report-1')
    with pytest.raises(ResearchError, match='under the private'):
        reporter.report(root, tmp_path / 'public-gallery')


def test_final_completion_recovery_preserves_actual_first_restore_history(tmp_path, monkeypatch):
    root = completed_fixture(tmp_path, monkeypatch)
    folder = root / 'A2-train'
    first_proof = load_json(folder / 'resume-verification.json')
    save_json(folder / 'resume-verification-0123456789abcdef.json', first_proof)
    final_proof = dict(first_proof, restored_step=200, remaining_updates=0, checkpoint_sha256=sha256(folder / 'last.pt'),
        adapter_optimizer_scaler_rng_restored=False, exact_adapter_optimizer_scaler_readback_verified=False,
        exact_python_numpy_torch_cpu_cuda_rng_readback_verified=False, completion_without_optimizer_updates=True)
    set_latest_proof(root, final_proof)
    result = reporter.report(root, root / 'completion-recovery-report')['startup_resume']
    assert result['latest_checkpoint_step'] == 200
    assert not result['latest_actual_restore_performed']
    assert result['latest_budget_complete_verification'] and result['actual_first_update_restore_proof_verified']
    assert result['retained_resume_proofs'] == 2


@pytest.mark.parametrize('problem', ['missing_update', 'unmatched_seed_schedule', 'wrong_coefficient', 'nonfinite_checkpoint',
    'bad_moment', 'bad_rng', 'missing_checkpoint', 'extra_prediction', 'wrong_target', 'changed_prediction', 'invalid_probability',
    'wrong_aggregate', 'changed_mask', 'wrong_checkpoint_step', 'wrong_generation_config', 'wrong_startup_receipt', 'wrong_resume_proof',
    'missing_startup_bytes', 'changed_startup_bytes', 'bad_startup_rng', 'bad_startup_moment', 'bad_startup_identity',
    'missing_readback_flag', 'false_readback_flag', 'fake_final_restore', 'lost_first_restore_proof', 'changed_archived_first_proof'])
def test_report_rejects_material_discrepancies_before_creating_output(tmp_path, monkeypatch, problem):
    root = completed_fixture(tmp_path, monkeypatch)
    if problem in ('missing_update', 'unmatched_seed_schedule', 'wrong_coefficient'):
        path = root / 'A3-train/training.jsonl'; log = read_jsonl(path)
        if problem == 'missing_update': log.pop()
        elif problem == 'unmatched_seed_schedule': log[0]['sampled_view_ids'].reverse()
        else: log[0]['surface_weight'] = 0.
        write_jsonl(path, log)
    elif problem in ('nonfinite_checkpoint', 'bad_moment', 'bad_rng'):
        path = root / 'A3-train/step-000050.pt'; state = torch.load(path, map_location='cpu', weights_only=True)
        if problem == 'nonfinite_checkpoint': next(iter(state['adapter'].values())).flatten()[0] = float('nan')
        elif problem == 'bad_moment': state['optimizer']['state'][0]['exp_avg_sq'].flatten()[0] = -1
        else: state['rng']['python'] = random.Random(100).getstate()
        torch.save(state, path)
    elif problem == 'missing_checkpoint': (root / 'A3-train/step-000100.pt').unlink()
    elif problem == 'extra_prediction': shutil.copyfile(next((root / 'A3-fitting').glob('*.png')), root / 'A3-fitting/SYNTHETIC-extra.png')
    elif problem == 'wrong_target':
        path = next((root / 'A3-eval/targets').glob('*.png')); Image.new('RGB', (256, 256), (200, 200, 200)).save(path)
    elif problem == 'changed_prediction':
        path = next((root / 'A3-fitting').glob('*.png')); path.write_bytes(path.read_bytes() + b'SYNTHETIC TAMPER')
    elif problem == 'invalid_probability':
        path = next((root / 'A3-eval/probabilities').glob('*.npy')); values = np.load(path); values[0, 0] = np.nan; np.save(path, values)
    elif problem == 'wrong_aggregate':
        path = root / 'A3-eval/summary.json'; value = load_json(path); value['group_weighted']['lpips'] = .01; save_json(path, value)
    elif problem == 'changed_mask':
        path = next((root / 'A3-eval/masks').glob('*.png')); values = np.asarray(Image.open(path)).copy(); values[60, 61] = 0; Image.fromarray(values).save(path)
    elif problem in ('wrong_checkpoint_step', 'wrong_generation_config'):
        path = root / 'A3-fitting/generation.json'; value = load_json(path)
        if problem == 'wrong_checkpoint_step': value['trained_steps'] = 150
        else: value['config']['surface_weight'] = .1
        save_json(path, value)
        summary_path = root / 'A3-eval/summary.json'; summary = load_json(summary_path)
        summary.update(generation=value, generation_sha256=sha256(path)); save_json(summary_path, summary)
    elif problem == 'wrong_startup_receipt':
        path = root / 'A2-train/startup-receipt.json'; value = load_json(path); value['first_log_row_sha256'] = 'wrong'; save_json(path, value)
    elif problem == 'wrong_resume_proof':
        path = root / 'A2-train/resume-verification.json'; value = load_json(path); value['remaining_updates'] = 198; save_json(path, value)
    elif problem == 'missing_startup_bytes': (root / 'A2-train/startup.pt').unlink()
    elif problem == 'changed_startup_bytes':
        path = root / 'A2-train/startup.pt'; path.write_bytes(path.read_bytes() + b'SYNTHETIC TAMPER')
    elif problem in ('bad_startup_rng', 'bad_startup_moment', 'bad_startup_identity'):
        path = root / 'A2-train/startup.pt'; value = torch.load(path, map_location='cpu', weights_only=True)
        if problem == 'bad_startup_rng': value['rng']['python'] = random.Random(100).getstate()
        elif problem == 'bad_startup_moment': value['optimizer']['state'][0]['exp_avg_sq'].flatten()[0] = -1
        else: value['identity_hash'] = 'wrong'
        torch.save(value, path); repin_synthetic_startup(root)
    elif problem in ('missing_readback_flag', 'false_readback_flag'):
        value = load_json(root / 'A2-train/resume-verification.json')
        if problem == 'missing_readback_flag': value.pop('exact_python_numpy_torch_cpu_cuda_rng_readback_verified')
        else: value['exact_adapter_optimizer_scaler_readback_verified'] = False
        set_latest_proof(root, value)
    else:
        folder = root / 'A2-train'; first = load_json(folder / 'resume-verification.json')
        if problem != 'lost_first_restore_proof':
            saved = dict(first)
            if problem == 'changed_archived_first_proof': saved['checkpoint_sha256'] = 'wrong'
            save_json(folder / 'resume-verification-0123456789abcdef.json', saved)
        value = dict(first, restored_step=200, remaining_updates=0, checkpoint_sha256=sha256(folder / 'last.pt'),
            adapter_optimizer_scaler_rng_restored=False, exact_adapter_optimizer_scaler_readback_verified=False,
            exact_python_numpy_torch_cpu_cuda_rng_readback_verified=False, completion_without_optimizer_updates=True)
        if problem == 'fake_final_restore': value['adapter_optimizer_scaler_rng_restored'] = True
        set_latest_proof(root, value)
    with pytest.raises((ResearchError, ValueError)):
        reporter.report(root, root / 'rejected-report')
    assert not (root / 'rejected-report').exists()
