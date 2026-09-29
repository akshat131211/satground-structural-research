"""Final-only CPU audit of the conditional saturated eight-view fitting pair.

Only aggregate.json is publishable. Galleries, raw predictions, masks, sample
identities and checkpoint/RNG payloads remain private. Co-primary full and
nonperimeter boundary criteria are fixed by the saturated trial protocol.
"""
import argparse
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

from audit_reviewed_fitting_outputs import checkpoint_payload, exact_files, rgb_pixels
from diagnose_fitting_boundaries import boundary_counts
from prepare_reviewed_fitting import check_pins, require, verify_bundle
from prepare_saturated_fitting import make_configs, verify_trial
from report_reviewed_fitting import compare, training_audit
from satground.adapter import StructuralAdapter
from satground.annotations import binary_mask
from satground.common import load_json, object_hash, provenance, read_jsonl, safe_data_path, save_json, sha256
from satground.metrics import building_metrics, grouped_mean, paired_group_bootstrap
from train_reviewed_fitting import finite_tree

STAGES = ['A0-fitting', 'A0-eval', 'A2-train', 'A2-fitting', 'A2-eval', 'A3-train', 'A3-fitting', 'A3-eval']
METRICS = ('boundary_error', 'nonperimeter_boundary_error', 'building_iou', 'lpips', 'ssim')


def paired_comparison(control, proposed):
    values = compare(control, proposed)
    a, b = {row['sample_id']: row for row in control}, {row['sample_id']: row for row in proposed}
    reference = grouped_mean(control, 'nonperimeter_boundary_error')
    change = reference - grouped_mean(proposed, 'nonperimeter_boundary_error')
    joint = sum(b[s]['boundary_error'] < a[s]['boundary_error'] and
                b[s]['nonperimeter_boundary_error'] < a[s]['nonperimeter_boundary_error'] for s in a)
    values.update(relative_nonperimeter_reduction=change / reference if reference > 0 else None,
                  absolute_nonperimeter_reduction=change,
                  jointly_improved_views=joint, compared_views=len(a), fraction_views_jointly_improved=joint / len(a))
    groups = len({row['geo_group'] for row in control})
    values['geographic_intervals'] = ({key: paired_group_bootstrap(control, proposed, key=key, repeats=2000, seed=17)
                                      for key in ('boundary_error', 'nonperimeter_boundary_error')}
                                     if groups >= 2 else None)
    values['interval_scope'] = 'descriptive training-example fitting only; no held-out inference'
    return values


def fitting_decision(rows, thresholds):
    require(set(rows) == {'A0', 'A2', 'A3'} and finite_tree(rows), 'Missing/nonfinite final fitting evidence.')
    comparisons = {name: paired_comparison(rows[name], rows['A3']) for name in ('A0', 'A2')}
    checks = {}
    for name, values in comparisons.items():
        checks[name + '_full_boundary'] = (values['relative_boundary_reduction'] is not None and
            values['relative_boundary_reduction'] >= thresholds['minimum_relative_boundary_reduction'])
        checks[name + '_nonperimeter_boundary'] = (values['relative_nonperimeter_reduction'] is not None and
            values['relative_nonperimeter_reduction'] >= thresholds['minimum_relative_nonperimeter_reduction'])
        checks[name + '_iou'] = values['iou_change'] >= thresholds['minimum_iou_change']
        checks[name + '_lpips'] = (values['relative_lpips_increase'] is not None and
            values['relative_lpips_increase'] <= thresholds['maximum_relative_lpips_increase'])
    checks['joint_fraction_views_improved'] = (comparisons['A2']['fraction_views_jointly_improved'] >=
                                             thresholds['minimum_fraction_views_jointly_improved'])
    return dict(status='fitting_response_only' if all(checks.values()) else 'fitting_response_not_established',
                checks=checks, comparisons=comparisons, co_primary_metrics=['boundary_error', 'nonperimeter_boundary_error'],
                checkpoint_selection='fixed final update200 only', server_gate_eligible=False,
                held_out_generalization_evidence=False)


def aggregate(rows):
    result = {}
    for name, selected in [('all', rows), ('target_building', [r for r in rows if r['target_building_pixels'] > 0]),
                           ('target_empty', [r for r in rows if r['target_building_pixels'] == 0])]:
        result[name] = dict(views=len(selected), metrics={key: grouped_mean(selected, key) for key in METRICS} if selected else None,
            false_positive_fraction=grouped_mean(selected, 'false_positive_fraction') if selected else None,
            zero_building_predictions=sum(r['predicted_building_pixels'] == 0 for r in selected),
            full_boundary_empty_counts={case: sum(r['full_boundary_empty_case'] == case for r in selected)
                                        for case in ('both_empty', 'one_empty', 'neither_empty')},
            nonperimeter_boundary_empty_counts={case: sum(r['nonperimeter_empty_case'] == case for r in selected)
                                                for case in ('both_empty', 'one_empty', 'neither_empty')},
            full_missing_or_extra_boundary_penalty_views=sum(r['full_boundary_empty_case'] == 'one_empty' for r in selected),
            nonperimeter_missing_or_extra_boundary_penalty_views=sum(r['nonperimeter_empty_case'] == 'one_empty' for r in selected),
            primary_target_edge_pixels=sum(r['primary_target_edge_pixels'] for r in selected),
            target_perimeter_edge_pixels=sum(r['target_perimeter_edge_pixels'] for r in selected),
            target_nonperimeter_edge_pixels=sum(r['target_nonperimeter_edge_pixels'] for r in selected),
            predicted_nonperimeter_edge_pixels=sum(r['predicted_nonperimeter_edge_pixels'] for r in selected))
    return result


def audit_training(root, name, expected_identity, protocol, manifest, config, trial_record):
    ids = {row['sample_id'] for row in manifest}
    log, result = training_audit(root, name, expected_identity, protocol, ids)
    run = load_json(root / f'{name}-train/run.json')
    identity = run['identity']
    require(identity['config'] == config and identity['pipeline_source_hash'] == trial_record['pipeline_source_hash'],
            'Training configuration or core source differs from the frozen trial.')
    require(identity['trial_sha256'] == sha256(Path(config['saturated_trial']) / 'trial.json')
            and identity['bundle_sha256'] == trial_record['original_bundle_sha256'], 'Training trial/bundle identity differs.')
    require(identity['bundle_inputs'] == trial_record['source_file_sha256']
            and identity['bundle_artifacts'] == trial_record['artifact_sha256'], 'Training source/input pin records differ.')
    require(bool(identity['source_file_sha256']) and all(trial_record['source_file_sha256'].get(path) == digest
            for path, digest in identity['source_file_sha256'].items()), 'Training source pins are not in the frozen trial.')
    check_pins(identity['source_file_sha256'])
    for key, path in [('fitting_manifest_sha256', config['fitting_manifest']),
                      ('parent_training_manifest_sha256', config['train_manifest']), ('style_sha256', config['style'])]:
        require(identity[key] == sha256(path), 'Training manifest/style identity differs.')
    required = {'rgb', 'lpips', 'region', 'boundary', 'boundary_weight', 'legacy_boundary_diagnostic',
                'surface', 'surface_weight', 'total', 'gradient_norm', 'elapsed_seconds', 'peak_vram_mib', 'sampled_view_ids', 'step'}
    sampler, prefix_rng = random.Random(protocol['seed']), {}
    for row in log:
        require(required <= set(row), 'Incomplete saturated fitting update fields.')
        expected = [manifest[sampler.randrange(len(manifest))]['sample_id'] for _ in range(protocol['accumulation'])]
        require(row['sampled_view_ids'] == expected, 'Training sample schedule differs from its fixed seed.')
        require(row['boundary'] == row['boundary_weight'] == 0 and row['surface_weight'] == config['surface_weight'],
                'Historical boundary entered the saturated trial objective.')
        total = row['rgb'] + config['perceptual_weight'] * row['lpips'] + config['region_weight'] * row['region'] + config['surface_weight'] * row['surface']
        require(math.isclose(total, row['total'], rel_tol=2e-6, abs_tol=2e-6), 'Logged structural objective coefficients differ.')
        require(row['gradient_norm'] >= 0 and row['elapsed_seconds'] >= 0 and row['peak_vram_mib'] >= 0,
                'Invalid gradient norm, memory or timing.')
        prefix_rng[row['step']] = sampler.getstate()
    with torch.random.fork_rng(devices=[]):
        shapes = {key: tuple(value.shape) for key, value in StructuralAdapter(64, 16).state_dict().items()}
    checkpoints = []
    for step in range(protocol['save_every'], protocol['steps'] + 1, protocol['save_every']):
        path = root / f'{name}-train/step-{step:06d}.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        checkpoint_payload(state, shapes, step, config)
        require(state['rng']['python'] == prefix_rng[step], 'Checkpoint sampler RNG differs from its exact training prefix.')
        require(state['fitting_manifest_sha256'] == identity['fitting_manifest_sha256']
                and state['train_manifest_sha256'] == identity['parent_training_manifest_sha256']
                and state['train_data_digest'] == identity['data_digest'], 'Checkpoint inputs differ from its training identity.')
        require(state['provenance']['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Checkpoint core provenance differs.')
        checkpoints.append(sha256(path))
    final = torch.load(root / f'{name}-train/last.pt', map_location='cpu', weights_only=True)
    checkpoint_payload(final, shapes, protocol['steps'], config)
    tails = sorted((root / f'{name}-train').glob('interrupted-steps-*.jsonl'))
    result.update(optimizer_and_rng_payloads_verified=True, trainable_parameters=4448,
                  periodic_checkpoint_set_sha256=object_hash(checkpoints),
                  final_checkpoint_sha256=sha256(root / f'{name}-train/last.pt'),
                  retained_interrupted_tail_rows=sum(len(read_jsonl(path)) for path in tails),
                  cuda_rng_check='recorded byte dtype/dimensions only; no CUDA restore performed by reporter')
    return log, result


def audit_startup(root, status, logs, protocol, manifest, config):
    """Audit retained first-update bytes and distinguish restore from completion.

    Readback flags are recorded evidence from the trainer's immediate check. This
    CPU reporter validates the retained payload; it does not restore CUDA state.
    A later final-checkpoint recovery must retain the earlier actual step1 proof.
    """
    folder = root / 'A2-train'
    path = folder / 'startup-receipt.json'
    receipt = load_json(path)
    required_receipt = {'step', 'status', 'identity_hash', 'budget_steps', 'remaining_updates',
                        'updates_included_in_budget', 'sample_prefix_verified', 'first_log_row_sha256',
                        'checkpoint_sha256', 'startup_checkpoint_sha256'}
    require(required_receipt <= set(receipt), 'Incomplete startup receipt.')
    startup_path = folder / 'startup.pt'
    require(startup_path.is_file(), 'Exact original startup checkpoint was not retained.')
    require(status['startup_update'] == receipt['step'] == 1 and receipt['status'] == 'startup_complete'
            and receipt['identity_hash'] == status['identity']['training_identity']['A2']
            and receipt['budget_steps'] == protocol['steps'] and receipt['remaining_updates'] == protocol['steps'] - 1
            and receipt['updates_included_in_budget'] is True and receipt['sample_prefix_verified'] is True
            and receipt['first_log_row_sha256'] == object_hash(logs['A2'][0])
            and receipt['checkpoint_sha256'] == receipt['startup_checkpoint_sha256']
            == status['startup_checkpoint_sha256'] == sha256(startup_path),
            'Startup receipt differs from the retained first update.')
    first = torch.load(startup_path, map_location='cpu', weights_only=True)
    identity = load_json(folder / 'run.json')['identity']
    require(first['step'] == 1 and first['identity_hash'] == receipt['identity_hash']
            and first['config'] == identity['config'] == config, 'Retained startup checkpoint identity differs.')
    with torch.random.fork_rng(devices=[]):
        shapes = {key: tuple(value.shape) for key, value in StructuralAdapter(64, 16).state_dict().items()}
    checkpoint_payload(first, shapes, 1, config)
    sampler = random.Random(protocol['seed'])
    prefix = [manifest[sampler.randrange(len(manifest))]['sample_id'] for _ in range(protocol['accumulation'])]
    require(prefix == logs['A2'][0]['sampled_view_ids'] and sampler.getstate() == first['rng']['python'],
            'Retained startup sampler RNG differs from its first update.')
    require(first['fitting_manifest_sha256'] == identity['fitting_manifest_sha256']
            and first['train_manifest_sha256'] == identity['parent_training_manifest_sha256']
            and first['train_data_digest'] == identity['data_digest']
            and first['provenance']['pipeline_source_hash'] == identity['pipeline_source_hash'],
            'Retained startup input or core provenance differs.')
    proof_path = folder / 'resume-verification.json'
    proof = load_json(proof_path)
    require(status['verified_resumes']['A2-train'] == proof, 'Latest resume proof differs from runner state.')
    archives = sorted(folder.glob('resume-verification-*.json'))
    proofs = [(proof_path, proof), *[(p, load_json(p)) for p in archives]]
    required_proof = {'identity_hash', 'restored_step', 'remaining_updates', 'checkpoint_sha256',
                      'sample_prefix_verified', 'checkpoint_payload_verified', 'adapter_optimizer_scaler_rng_restored',
                      'exact_adapter_optimizer_scaler_readback_verified',
                      'exact_python_numpy_torch_cpu_cuda_rng_readback_verified',
                      'completion_without_optimizer_updates', 'optimizer_updates_performed'}
    actual_first = []
    for proof_file, value in proofs:
        require(required_proof <= set(value), 'Incomplete retained resume proof.')
        step = value['restored_step']
        require(type(step) is int and 1 <= step <= protocol['steps']
                and value['identity_hash'] == receipt['identity_hash']
                and value['remaining_updates'] == protocol['steps'] - step
                and value['sample_prefix_verified'] is True and value['checkpoint_payload_verified'] is True
                and type(value['optimizer_updates_performed']) is int and value['optimizer_updates_performed'] == 0,
                'Compatible resume proof has a different identity, sample prefix or budget.')
        restored = step < protocol['steps']
        require(all(value[key] is restored for key in ('adapter_optimizer_scaler_rng_restored',
                    'exact_adapter_optimizer_scaler_readback_verified',
                    'exact_python_numpy_torch_cpu_cuda_rng_readback_verified'))
                and value['completion_without_optimizer_updates'] is (not restored),
                'Resume proof does not distinguish actual restoration from budget-complete verification.')
        if step == 1:
            require(value['checkpoint_sha256'] == sha256(startup_path), 'First-update resume restored different checkpoint bytes.')
            actual_first.append(sha256(proof_file))
        elif step % protocol['save_every'] == 0:
            require(value['checkpoint_sha256'] == sha256(folder / f'step-{step:06d}.pt'),
                    'Periodic checkpoint resume proof differs from retained checkpoint bytes.')
        else:
            require(False, 'Resume proof refers to a checkpoint outside the fixed retained schedule.')
    require(bool(actual_first), 'Actual first-update restore proof was not retained in latest proof or history.')
    return dict(first_update_included_in_budget=True, first_log_receipt_verified=True,
                retained_resume_proof_verified=True, latest_checkpoint_step=proof['restored_step'],
                latest_actual_restore_performed=proof['adapter_optimizer_scaler_rng_restored'],
                latest_budget_complete_verification=proof['completion_without_optimizer_updates'],
                actual_first_update_restore_proof_verified=True, retained_resume_proofs=len(proofs),
                original_startup_checkpoint_bytes_reinspected=True, startup_payload_verified=True,
                startup_receipt_sha256=sha256(path), startup_checkpoint_sha256=sha256(startup_path),
                resume_proof_sha256=sha256(proof_path), retained_resume_proof_set_sha256=object_hash(sorted(sha256(p) for p, _ in proofs)),
                actual_first_update_restore_proof_set_sha256=object_hash(sorted(actual_first)),
                restore_readback_evidence='trainer recorded exact immediate adapter/optimizer/scaler and Python/NumPy/TorchCPU/CUDA readback; reporter CPU payload audit only')


def report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'Use fresh local report output under the private saturated run.')
    status = load_json(root / 'status.json')
    require(status['status'] in ('reporting', 'complete') and status['completed'] == STAGES
            and status.get('child_pid') is None, 'Serial saturated stages incomplete or a model child remains live.')
    identity = status['identity']
    check_pins(identity['files'])
    record = verify_trial(identity['trial'])
    require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'] == record['pipeline_source_hash'],
            'Core source differs from frozen saturated trial.')
    bundle = Path(identity['bundle']).resolve()
    require(bundle == Path(record['original_bundle']).resolve()
            and sha256(bundle / 'bundle.json') == identity['bundle_sha256'] == record['original_bundle_sha256'], 'Original fitting bundle differs.')
    original = verify_bundle(bundle)
    protocol = record['protocol']
    require(protocol['steps'] == 200 and protocol['save_every'] == 50 and protocol['accumulation'] == 8,
            'Saturated report requires the fixed final200 budget.')
    manifest = read_jsonl(bundle / 'manifest.jsonl')
    manual = load_json(bundle / 'manual-index.json')
    ids = {row['sample_id'] for row in manifest}
    require(len(manifest) == len(ids) == len(manual) == original['fitting_count'] == 8 and set(manual) == ids
            and all(row['split'] == 'train' and row['city'] in ('Chicago', 'NewYork', 'SanFrancisco') for row in manifest),
            'Expected unchanged eight training-only fitting views.')
    configs = make_configs(identity['trial'], protocol, bundle)
    logs, training, rows, evidence = {}, {}, {}, {}
    for name in ('A2', 'A3'):
        logs[name], training[name] = audit_training(root, name, identity['training_identity'][name], protocol, manifest, configs[name], record)
    require([row['sampled_view_ids'] for row in logs['A2']] == [row['sampled_view_ids'] for row in logs['A3']], 'A2/A3 sampled schedules differ.')
    startup = audit_startup(root, status, logs, protocol, manifest, configs['A2'])
    image_names, probability_names = {sid + '.png' for sid in ids}, {sid + '.npy' for sid in ids}
    labels_reference, generation_reference = None, None
    for name in ('A0', 'A2', 'A3'):
        images, evaluation = root / f'{name}-fitting', root / f'{name}-eval'
        exact_files(images, image_names | {'generation.json', 'predictions.jsonl'})
        for subdir in ('masks', 'targets'):
            exact_files(evaluation / subdir, image_names)
        exact_files(evaluation / 'probabilities', probability_names)
        summary, generation = load_json(evaluation / 'summary.json'), load_json(images / 'generation.json')
        cfg = generation['config']
        require(cfg == configs[name], 'Generation configuration differs from the frozen saturated config.')
        require(summary['generation'] == generation and summary['generation_sha256'] == sha256(images / 'generation.json'), 'Evaluation/generation metadata differs.')
        require(generation['manifest_sha256'] == summary['manifest_sha256'] == original['fitting_manifest_sha256']
                and generation['scientific'] and generation['target_images_used_for_conditioning'] is False
                and generation['stress'] == 'clean', 'Incompatible generation inputs or target conditioning.')
        require(generation['style_sha256'] == sha256(cfg['style'])
                and generation['train_manifest_sha256'] == original['parent_training_manifest_sha256'], 'Fixed training-derived illumination differs.')
        require(summary['manual_index_sha256'] == sha256(bundle / 'manual-index.json')
                and summary['segmenter_revision'] == protocol['evaluation_segmenter_revision']
                and summary['training_segmenter_different'] is True and summary['manually_reviewed_count'] == len(ids)
                and summary['count'] == len(ids), 'Evaluation teacher or approved scoring masks differ.')
        for metadata in (generation['provenance'], summary['evaluation_provenance']):
            require(metadata['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Renderer/evaluator core source differs.')
        if generation_reference is not None:
            require(all(summary[key] == generation_reference[key] for key in ('segmenter', 'segmenter_revision', 'empty_mask_convention')),
                    'Paired evaluator conventions differ.')
        generation_reference = summary
        require(generation['trained_steps'] == (0 if name == 'A0' else protocol['steps']), 'Wrong final-only scoring step.')
        if name == 'A0':
            require(generation['checkpoint_sha256'] is None and generation.get('training_identity_hash') is None, 'Frozen reference carries a trained adapter.')
        else:
            require(generation['checkpoint_sha256'] == sha256(root / f'{name}-train/last.pt')
                    and generation['training_identity_hash'] == identity['training_identity'][name], 'Scored final checkpoint identity differs.')
        predictions = read_jsonl(images / 'predictions.jsonl')
        require(len(predictions) == len(ids) and {row['sample_id'] for row in predictions} == ids, 'Missing/duplicate/extra predictions.')
        for prediction in predictions:
            require(prediction['prediction'] == prediction['sample_id'] + '.png'
                    and sha256(safe_data_path(images, prediction['prediction'])) == prediction['sha256'], 'Prediction bytes or image path changed.')
        metrics = read_jsonl(evaluation / 'metrics.jsonl')
        require(len(metrics) == len(ids) and {row['sample_id'] for row in metrics} == ids and finite_tree(metrics)
                and all(row['split'] == 'train' and row['label_provenance'] == 'manually_reviewed' for row in metrics), 'Incomplete reviewed fitting-only scores.')
        by_manifest = {row['sample_id']: row for row in manifest}
        require(all(row['geo_group'] == by_manifest[row['sample_id']]['geo_group'] for row in metrics), 'Scored geographic labels differ from frozen manifest.')
        label_identity = {row['sample_id']: (row['geo_group'], row['valid_pixels'], row['target_building_pixels']) for row in metrics}
        require(labels_reference is None or label_identity == labels_reference, 'Paired scoring target labels differ.')
        labels_reference = label_identity
        artifact_hashes = {'predictions': {}, 'masks': {}, 'targets': {}, 'probabilities': {}}
        for row in metrics:
            sid, entry = row['sample_id'], manual[row['sample_id']]
            rgb_pixels(images / (sid + '.png'), protocol['size'])
            target_path = evaluation / 'targets' / (sid + '.png')
            target_rgb = rgb_pixels(target_path, protocol['size'])
            require(sha256(target_path) == entry['target_image_sha256']
                    and np.array_equal(target_rgb, rgb_pixels(entry['target_image'], protocol['size'])), 'Target bytes/pixels differ from approved crop.')
            shape = (protocol['size'], protocol['size'])
            predicted = binary_mask(evaluation / 'masks' / (sid + '.png'), shape)
            target, valid = binary_mask(entry['building_mask'], shape), binary_mask(entry['valid_mask'], shape)
            actual = building_metrics(predicted, target, valid)
            require(all(actual[key] == row[key] for key in actual), 'Full structural metrics do not reproduce from unchanged masks.')
            counts = boundary_counts(predicted, target, valid)
            row.update(counts, false_positive_fraction=float((predicted & ~target & valid).sum() / valid.sum()))
            pe, te = counts['primary_predicted_edge_pixels'], counts['primary_target_edge_pixels']
            row['full_boundary_empty_case'] = 'both_empty' if not pe and not te else 'one_empty' if not pe or not te else 'neither_empty'
            probability_path = evaluation / 'probabilities' / (sid + '.npy')
            values = np.load(probability_path, allow_pickle=False)
            require(values.shape == shape and values.dtype == np.float32 and np.isfinite(values).all()
                    and ((values >= 0) & (values <= 1)).all(), 'Invalid evaluator probability array.')
            for key, path in [('predictions', images / (sid + '.png')), ('masks', evaluation / 'masks' / (sid + '.png')),
                              ('targets', target_path), ('probabilities', probability_path)]:
                artifact_hashes[key][sid] = sha256(path)
        for key, value in summary['group_weighted'].items():
            require(key in METRICS and math.isclose(grouped_mean(metrics, key), value, rel_tol=0, abs_tol=1e-12), 'Evaluation aggregate differs from per-view scores.')
        require(set(summary['group_weighted']) == {'boundary_error', 'building_iou', 'lpips', 'ssim'}, 'Missing/wrong original evaluation aggregates.')
        rows[name] = metrics
        evidence[name] = {key: sha256(path) for key, path in {
            'metrics': evaluation / 'metrics.jsonl', 'summary': evaluation / 'summary.json',
            'generation': images / 'generation.json', 'prediction_index': images / 'predictions.jsonl'}.items()}
        evidence[name].update({key + '_artifact_set_sha256': object_hash(value) for key, value in artifact_hashes.items()})
    require(finite_tree(rows), 'Recomputed final metrics are nonfinite.')
    result = dict(scope='training_fitting_only', method='valid_surface_saturated_v2', fitted_views=8,
        submitted_views=original['submitted_count'], excluded_views=original['excluded_count'],
        historical_positive_contour_pixels=original['positive_contour_pixels'],
        decision=fitting_decision(rows, protocol['decision']), models={name: aggregate(values) for name, values in rows.items()},
        training=training, startup_resume=startup, evidence_sha256=evidence, matched_sample_schedule=True,
        completion_audit=dict(training_rows_verified=400, periodic_checkpoints_verified=8, last_checkpoints_verified=2,
            rgb_images_verified=24, evaluated_masks_verified=24, reviewed_target_copies_verified=24,
            probability_arrays_verified=24, optimizer_and_rng_payloads_verified=True,
            probability_hash_origin='completion snapshot; original evaluator does not independently record NPY hashes',
            cuda_operations=0, original_startup_checkpoint_bytes_reinspected=True),
        co_primary_metric_conventions=dict(full='unchanged building_metrics inner edges including crop-closing edges',
            nonperimeter='unchanged boundary_counts: remove outermost rows/columns from both edge sets only; both-empty0, one-empty1'),
        geographic_interval_scope='descriptive fitting only; eight reviewed training examples and one seed',
        final_only_evaluation=True, checkpoint_selection_performed=False, no_held_out_evaluation=True,
        sealed_images_opened=False, server_gate_eligible=False, held_out_generalization_evidence=False,
        annotation_accuracy_independently_certified=False, manual_gallery_review_pending=True,
        source_and_input_pins_verified=True, reporter_sha256=sha256(__file__))
    output.mkdir(parents=True)
    save_json(output / 'aggregate.json', result)
    size, stride = protocol['size'], protocol['size'] + 54
    canvas = Image.new('RGB', (size * 4, len(manifest) * stride), 'white')
    draw = ImageDraw.Draw(canvas)
    indices = {name: {row['sample_id']: row for row in values} for name, values in rows.items()}
    for i, row in enumerate(manifest):
        sid, y = row['sample_id'], i * stride
        paths = [manual[sid]['target_image'], *[root / f'{name}-fitting' / f'{sid}.png' for name in ('A0', 'A2', 'A3')]]
        for col, path in enumerate(paths):
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB'), (col * size, y))
            if col == 0:
                draw.text((col * size + 4, y + size + 3), f'Fitting view {i + 1} | target', fill='black')
            else:
                name = ('A0', 'A2', 'A3')[col - 1]
                score = indices[name][sid]
                draw.text((col * size + 4, y + size + 3), f'{name} | final' if name != 'A0' else 'A0 | frozen', fill='black')
                draw.text((col * size + 4, y + size + 17), f"full={score['boundary_error']:.4f} nonperi={score['nonperimeter_boundary_error']:.4f}", fill='black')
                draw.text((col * size + 4, y + size + 31), f"IoU={score['building_iou']:.3f} LPIPS={score['lpips']:.3f}", fill='black')
    canvas.save(output / 'paired-gallery.png')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.run, args.output)
    print(result['decision']['status'])
