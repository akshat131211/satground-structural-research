"""Verify a completed fitting diagnosis and produce aggregate-only conclusions."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from prepare_reviewed_fitting import check_pins, make_configs, require, verify_bundle
from satground.annotations import binary_mask
from satground.common import load_json, object_hash, provenance, read_jsonl, safe_data_path, save_json, sha256
from satground.metrics import building_metrics, grouped_mean
from train_reviewed_fitting import finite_tree

METRICS = ('boundary_error', 'building_iou', 'lpips', 'ssim')


def compare(a, b):
    require(a and len(a) == len(b), 'Incomplete paired fitting metrics.')
    am, bm = {r['sample_id']: r for r in a}, {r['sample_id']: r for r in b}
    require(len(am) == len(a) and set(am) == set(bm), 'Duplicate or unmatched fitting predictions.')
    require(all(am[s]['geo_group'] == bm[s]['geo_group'] for s in am), 'Fitting groups differ.')
    boundary, lpips = grouped_mean(a, 'boundary_error'), grouped_mean(a, 'lpips')
    return dict(relative_boundary_reduction=(boundary - grouped_mean(b, 'boundary_error')) / boundary if boundary > 0 else None,
                absolute_boundary_reduction=boundary - grouped_mean(b, 'boundary_error'),
                iou_change=grouped_mean(b, 'building_iou') - grouped_mean(a, 'building_iou'),
                relative_lpips_increase=(grouped_mean(b, 'lpips') - lpips) / lpips if lpips > 0 else None,
                fraction_views_boundary_improved=sum(bm[s]['boundary_error'] < am[s]['boundary_error'] for s in am) / len(am))


def fitting_decision(rows, thresholds):
    require(set(rows) == {'A0', 'A2', 'A3'} and finite_tree(rows), 'Missing/nonfinite fitting evidence.')
    comparisons = {name: compare(rows[name], rows['A3']) for name in ('A0', 'A2')}
    checks = {}
    for name, values in comparisons.items():
        checks[name + '_boundary'] = (values['relative_boundary_reduction'] is not None and
            values['relative_boundary_reduction'] >= thresholds['minimum_relative_boundary_reduction'])
        checks[name + '_iou'] = values['iou_change'] >= thresholds['minimum_iou_change']
        checks[name + '_lpips'] = (values['relative_lpips_increase'] is not None and
            values['relative_lpips_increase'] <= thresholds['maximum_relative_lpips_increase'])
    checks['fraction_views_improved'] = (comparisons['A2']['fraction_views_boundary_improved'] >=
                                        thresholds['minimum_fraction_views_boundary_improved'])
    return dict(status='fitting_response_only' if all(checks.values()) else 'fitting_response_not_established',
                checks=checks, comparisons=comparisons, server_gate_eligible=False, held_out_generalization_evidence=False)


def aggregate(rows):
    output = {}
    for label, subset in [('all', rows), ('target_building', [r for r in rows if r['target_building_pixels'] > 0]),
                          ('target_empty', [r for r in rows if r['target_building_pixels'] == 0])]:
        output[label] = dict(views=len(subset),
            metrics={m: grouped_mean(subset, m) for m in METRICS} if subset else None,
            false_positive_fraction=grouped_mean(subset, 'false_positive_fraction') if subset else None,
            maximum_boundary_penalty_views=sum(r['boundary_error'] == 1 for r in subset),
            zero_building_predictions=sum(r['predicted_building_pixels'] == 0 for r in subset))
    return output


def training_audit(root, name, expected_identity, protocol, allowed_ids):
    folder = root / f'{name}-train'
    run = load_json(folder / 'run.json')
    require(run['identity_hash'] == expected_identity == object_hash(run['identity']), 'Training identity differs from runner.')
    log = read_jsonl(folder / 'training.jsonl')
    steps = protocol['steps']
    require([r['step'] for r in log] == list(range(1, steps + 1)) and finite_tree(log), 'Wrong or nonfinite update log.')
    for row in log:
        require(len(row['sampled_view_ids']) == protocol['accumulation'] and set(row['sampled_view_ids']) <= allowed_ids,
                'Update used a view outside the resolved fitting subset.')
    count = 0
    expected_files = {f'step-{step:06d}.pt' for step in range(protocol['save_every'], steps + 1, protocol['save_every'])}
    require({p.name for p in folder.glob('step-*.pt')} == expected_files, 'Unexpected periodic checkpoint set.')
    for step in range(protocol['save_every'], steps + 1, protocol['save_every']):
        path = folder / f'step-{step:06d}.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(state['step'] == step and state['identity_hash'] == expected_identity and finite_tree(state), 'Invalid periodic checkpoint.')
        require(state['config'] == run['identity']['config'] and bool(state['adapter']) and bool(state['optimizer']['state']),
                'Checkpoint lacks the expected adapter or optimizer.')
        require(all(float(v['step']) == step for v in state['optimizer']['state'].values()), 'Optimizer update count differs.')
        count += 1
    require(sha256(folder / 'last.pt') == sha256(folder / f'step-{steps:06d}.pt'), 'Last checkpoint differs from final periodic checkpoint.')
    summary = load_json(folder / 'summary.json')
    require(summary['status'] == 'budget_complete' and summary['step'] == steps
            and summary['checkpoint_sha256'] == sha256(folder / 'last.pt'), 'Incomplete fitting summary.')
    return log, dict(updates=len(log), finite_losses=True, periodic_checkpoints=count,
                     peak_vram_mib=max(r['peak_vram_mib'] for r in log), elapsed_seconds=log[-1]['elapsed_seconds'])


def report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'Use fresh local report output under the private run.')
    status = load_json(root / 'status.json')
    expected_stages = ['A0-fitting', 'A0-eval', 'A2-train', 'A2-fitting', 'A2-eval', 'A3-train', 'A3-fitting', 'A3-eval']
    require(status['status'] in ('reporting', 'complete') and status['completed'] == expected_stages, 'Serial fitting stages incomplete.')
    identity = status['identity']
    check_pins(identity['files'])
    require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source differs from frozen fitting source.')
    bundle = Path(identity['bundle'])
    require(sha256(bundle / 'bundle.json') == identity['bundle_sha256'], 'Fitting bundle changed.')
    record = verify_bundle(bundle)
    protocol = record['protocol']
    manifest = read_jsonl(bundle / 'manifest.jsonl')
    manual = load_json(bundle / 'manual-index.json')
    ids = {r['sample_id'] for r in manifest}
    rows, training, logs, evidence = {}, {}, {}, {}
    for name in ('A2', 'A3'):
        logs[name], training[name] = training_audit(root, name, identity['training_identity'][name], protocol, ids)
    require([r['sampled_view_ids'] for r in logs['A2']] == [r['sampled_view_ids'] for r in logs['A3']],
            'A2/A3 sampled training schedules differ.')
    reference_labels = None
    for name in ('A0', 'A2', 'A3'):
        images, evaluation = root / f'{name}-fitting', root / f'{name}-eval'
        summary = load_json(evaluation / 'summary.json')
        generation = load_json(images / 'generation.json')
        cfg = generation['config']
        require(cfg == make_configs(bundle, protocol, cfg['train_manifest'], cfg['style'])[name], 'Unexpected generation configuration.')
        require(summary['generation'] == generation and summary['generation_sha256'] == sha256(images / 'generation.json'),
                'Evaluation/generation record differs.')
        require(generation['manifest_sha256'] == record['fitting_manifest_sha256'] and generation['scientific']
                and generation['target_images_used_for_conditioning'] is False and generation['stress'] == 'clean',
                'Generation uses incompatible inputs or target conditioning.')
        require(generation['style_sha256'] == sha256(cfg['style']) and
                generation['train_manifest_sha256'] == record['parent_training_manifest_sha256'], 'Illumination source differs.')
        require(summary['manual_index_sha256'] == sha256(bundle / 'manual-index.json')
                and summary['segmenter_revision'] == protocol['evaluation_segmenter_revision']
                and summary['training_segmenter_different'] and summary['manually_reviewed_count'] == len(ids),
                'Evaluator or reviewed masks differ.')
        for p in (generation['provenance'], summary['evaluation_provenance']):
            require(p['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Evaluator/renderer source differs.')
        require(generation['trained_steps'] == (0 if name == 'A0' else protocol['steps']), 'Wrong scoring checkpoint.')
        if name != 'A0':
            require(generation['checkpoint_sha256'] == sha256(root / f'{name}-train/last.pt')
                    and generation['training_identity_hash'] == identity['training_identity'][name], 'Scored checkpoint identity differs.')
        else:
            require(generation['checkpoint_sha256'] is None, 'Frozen reference has an adapter.')
        predictions = read_jsonl(images / 'predictions.jsonl')
        require(len(predictions) == len(ids) and {r['sample_id'] for r in predictions} == ids, 'Incomplete prediction set.')
        for pred in predictions:
            require(sha256(safe_data_path(images, pred['prediction'])) == pred['sha256'], 'Prediction bytes changed.')
        metrics = read_jsonl(evaluation / 'metrics.jsonl')
        require(len(metrics) == len(ids) and {r['sample_id'] for r in metrics} == ids and finite_tree(metrics)
                and all(r['split'] == 'train' and r['label_provenance'] == 'manually_reviewed' for r in metrics),
                'Incomplete reviewed training-only scores.')
        labels = {r['sample_id']: (r['geo_group'], r['valid_pixels'], r['target_building_pixels']) for r in metrics}
        require(reference_labels is None or labels == reference_labels, 'Paired scoring labels differ.')
        reference_labels = labels
        for row in metrics:
            entry = manual[row['sample_id']]
            shape = (protocol['size'], protocol['size'])
            predicted = binary_mask(evaluation / 'masks' / f"{row['sample_id']}.png", shape)
            target, valid = binary_mask(entry['building_mask'], shape), binary_mask(entry['valid_mask'], shape)
            actual = building_metrics(predicted, target, valid)
            require(all(actual[k] == row[k] for k in actual), 'Structural metrics do not reproduce from masks.')
            row['false_positive_fraction'] = float((predicted & ~target & valid).sum() / valid.sum())
        for key, value in summary['group_weighted'].items():
            require(abs(grouped_mean(metrics, key) - value) < 1e-12, 'Fitting aggregate mismatch.')
        rows[name] = metrics
        evidence[name] = {key: sha256(path) for key, path in {
            'metrics': evaluation / 'metrics.jsonl', 'summary': evaluation / 'summary.json',
            'generation': images / 'generation.json', 'predictions': images / 'predictions.jsonl'}.items()}
    result = dict(scope='training_fitting_only', fitted_views=len(ids), submitted_views=record['submitted_count'],
                  excluded_views=record['excluded_count'], positive_contour_pixels=record['positive_contour_pixels'],
                  decision=fitting_decision(rows, protocol['decision']), models={n: aggregate(v) for n, v in rows.items()},
                  training=training, evidence_sha256=evidence, matched_sample_schedule=True,
                  no_held_out_evaluation=True, sealed_images_opened=False, server_gate_eligible=False,
                  annotation_accuracy_independently_certified=False, manual_gallery_review_pending=True)
    output.mkdir()
    # Only this aggregate file is a publication candidate. All galleries remain local.
    save_json(output / 'aggregate.json', result)
    canvas = Image.new('RGB', (256 * 4, len(manifest) * 280), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(manifest):
        paths = [manual[row['sample_id']]['target_image'],
                 *[root / f'{name}-fitting' / f"{row['sample_id']}.png" for name in ('A0', 'A2', 'A3')]]
        for col, path in enumerate(paths):
            with Image.open(path) as image:
                canvas.paste(image.convert('RGB'), (col * 256, i * 280))
        draw.text((4, i * 280 + 258), f'Fitting view {i + 1}: target | frozen A0 | A2 | A3', fill='black')
    canvas.save(output / 'paired-gallery.png')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report(args.run, args.output)
