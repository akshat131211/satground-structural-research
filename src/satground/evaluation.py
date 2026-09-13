from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .camera import crop_panorama
from .common import (ResearchError, load_json, read_jsonl, require_development, safe_data_path,
                     save_json, sha256, write_jsonl)
from .metrics import building_metrics, grouped_mean, paired_group_bootstrap, server_gate
from .semantics import EVAL_SEGMENTER, TRAIN_SEGMENTER, Segmenter, rgb_tensor


def evaluate(manifest, predictions, data_root, output, model_id=EVAL_SEGMENTER, revision=None,
             manual_index=None, kid=False):
    rows = read_jsonl(manifest)
    require_development(rows)
    generation = load_json(Path(predictions) / 'generation.json')
    if generation.get('manifest_sha256') != sha256(manifest) or not generation.get('scientific'):
        raise ResearchError('Generation manifest mismatch or synthetic outputs supplied for scientific scoring.')
    if model_id == TRAIN_SEGMENTER:
        raise ResearchError('The evaluation segmenter must differ from the training segmenter.')
    predicted_index = {r['sample_id']: r for r in read_jsonl(Path(predictions) / 'predictions.jsonl')}
    if set(predicted_index) != {r['sample_id'] for r in rows}:
        raise ResearchError('Missing or extra predictions; paired evaluation must be complete.')
    manual = load_json(manual_index) if manual_index else {}
    segmenter = Segmenter(model_id, revision)
    import lpips
    perceptual = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for directory in ('targets', 'probabilities', 'masks'):
        (out / directory).mkdir(exist_ok=True)
    results = []
    for row in rows:
        sid = row['sample_id']
        pred_path = safe_data_path(predictions, predicted_index[sid]['prediction'])
        if sha256(pred_path) != predicted_index[sid]['sha256']:
            raise ResearchError('Prediction image was altered after generation.')
        predicted = np.asarray(Image.open(pred_path).convert('RGB'))
        target = crop_panorama(Image.open(safe_data_path(data_root, row['panorama'])).convert('RGB'),
                               row['yaw'], row['pitch'], row['fov'], row['size'])
        if target.shape != predicted.shape:
            raise ResearchError('Prediction and target have different shapes.')
        with torch.no_grad():
            p, t = rgb_tensor(predicted, 'cuda'), rgb_tensor(target, 'cuda')
            probabilities = segmenter.probabilities(p)
            target_probabilities = segmenter.probabilities(t)
            p_label = probabilities.argmax(1)[0].cpu().numpy()
            certainty, t_label = target_probabilities.max(1)
            certainty, t_label = certainty[0].cpu().numpy(), t_label[0].cpu().numpy()
            perceptual_value = float(perceptual(p * 2 - 1, t * 2 - 1).mean())
        valid = (certainty >= .7) & (t_label < 11)
        provenance = 'independent_segmenter_pseudo_label'
        target_mask = t_label == 2
        if sid in manual:
            entry = manual[sid]
            if entry.get('reviewed') is not True or not entry.get('reviewer') or not entry.get('reviewed_utc'):
                raise ResearchError('Manual mask lacks explicit reviewer, date, and completed review.')
            target_mask = np.asarray(Image.open(entry['building_mask']).convert('L')) >= 128
            valid = np.asarray(Image.open(entry['valid_mask']).convert('L')) >= 128
            if target_mask.shape != t_label.shape or valid.shape != t_label.shape:
                raise ResearchError('Manual masks must match the exact target crop.')
            provenance = 'manually_reviewed'
        metric = building_metrics(p_label == 2, target_mask, valid)
        psnr = float(peak_signal_noise_ratio(target, predicted, data_range=255))
        building_probability = probabilities[0, 2].cpu().numpy().astype(np.float32)
        np.save(out / 'probabilities' / f'{sid}.npy', building_probability)
        Image.fromarray(target).save(out / 'targets' / f'{sid}.png')
        Image.fromarray((p_label == 2).astype(np.uint8) * 255).save(out / 'masks' / f'{sid}.png')
        entropy = -(building_probability * np.log(building_probability + 1e-8) +
                    (1 - building_probability) * np.log(1 - building_probability + 1e-8))
        results.append(dict(sample_id=sid, geo_group=row['geo_group'], split=row['split'],
                            label_provenance=provenance, **metric, lpips=perceptual_value,
                            instance_errors=manual.get(sid, {}).get('instance_errors'),
                            psnr=psnr if np.isfinite(psnr) else None, identical_rgb=bool(np.array_equal(predicted, target)),
                            ssim=float(structural_similarity(target, predicted, channel_axis=2, data_range=255)),
                            entropy=float(entropy.mean())))
    write_jsonl(out / 'metrics.jsonl', results)
    summary = dict(count=len(results), manifest_sha256=sha256(manifest), generation_sha256=sha256(Path(predictions) / 'generation.json'),
                    generation=generation,
                    segmenter=model_id, segmenter_revision=segmenter.revision, training_segmenter_different=True,
                    manually_reviewed_count=sum(r['label_provenance'] == 'manually_reviewed' for r in results),
                    group_weighted={k: grouped_mean(results, k) for k in ('building_iou', 'boundary_error', 'lpips', 'ssim')},
                    kid=None, fid=None, empty_mask_convention='both_empty: IoU1,error0; one_empty: error1')
    if kid:
        if len(rows) < 100:
            raise ResearchError('KID requires at least 100 matched images in this pipeline.')
        from cleanfid import fid
        # Repeated subsets provide a dispersion estimate; not independent geographic confidence bounds.
        estimates = []
        for seed in (17, 29, 43, 61, 79):
            np.random.seed(seed)
            estimates.append(float(fid.compute_kid(str(predictions), str(out / 'targets'), device=torch.device('cuda'),
                                                   num_workers=0, batch_size=8)))
        summary['kid'] = dict(mean=float(np.mean(estimates)), subset_repeat_std=float(np.std(estimates, ddof=1)),
                              estimates=estimates, uncertainty_kind='subset_randomness_not_geographic_CI')
    save_json(out / 'summary.json', summary)
    failure_gallery(rows, results, predictions, out)
    return summary


def select_checkpoint(candidates_path, reference_summary, output):
    """Validation-only selection; audit results are rejected before metric inspection."""
    specification = load_json(candidates_path)
    reference = load_json(reference_summary)
    reference_lpips = reference['group_weighted']['lpips']
    candidates = []
    for candidate in specification['candidates']:
        summary = load_json(candidate['summary'])
        rows = read_jsonl(Path(candidate['summary']).parent / 'metrics.jsonl')
        if not rows or any(r['split'] != 'validation' for r in rows):
            raise ResearchError('Checkpoint selection may only use validation metrics.')
        if summary['manifest_sha256'] != reference['manifest_sha256']:
            raise ResearchError('Checkpoint selection uses mismatched validation manifests.')
        if summary['group_weighted']['lpips'] <= reference_lpips * 1.05:
            candidates.append((summary['group_weighted']['boundary_error'], candidate))
    if not candidates:
        raise ResearchError('No candidate satisfies the validation image-quality constraint.')
    selected = min(candidates, key=lambda c: (c[0], c[1]['checkpoint']))[1]
    result = dict(selected=selected, criterion='minimum_group_weighted_boundary_error_with_LPIPS_within_5_percent',
                  reference_sha256=sha256(reference_summary), candidate_spec_sha256=sha256(candidates_path))
    save_json(output, result)
    return result


def failure_gallery(rows, metrics, predictions, output, limit=24):
    rows = {r['sample_id']: r for r in rows}
    chosen = sorted(metrics, key=lambda r: (-r['boundary_error'], r['sample_id']))[:limit]
    size = 128
    canvas = Image.new('RGB', (size * 3, len(chosen) * (size + 24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, metric in enumerate(chosen):
        sid = metric['sample_id']
        y = i * (size + 24)
        for col, path in enumerate([Path(output) / 'targets' / f'{sid}.png', Path(predictions) / f'{sid}.png',
                                    Path(output) / 'masks' / f'{sid}.png']):
            canvas.paste(Image.open(path).convert('RGB').resize((size, size)), (col * size, y))
        draw.text((3, y + size + 3), f"{sid[:10]} boundary={metric['boundary_error']:.4f} | target / generated / building", fill='black')
    canvas.save(Path(output) / 'failure-gallery.png')


def comparison(control_path, proposed_path, seed):
    a, b = read_jsonl(control_path), read_jsonl(proposed_path)
    ci = paired_group_bootstrap(a, b)
    av = grouped_mean(a, 'boundary_error')
    al = grouped_mean(a, 'lpips')
    if av <= 0 or al <= 0:
        raise ResearchError('Relative improvement is undefined for a zero baseline error.')
    return dict(seed=seed, relative_boundary_reduction=(av - grouped_mean(b, 'boundary_error')) / av,
                boundary_ci95=ci['ci95'], iou_change=grouped_mean(b, 'building_iou') - grouped_mean(a, 'building_iou'),
                relative_lpips_change=(grouped_mean(b, 'lpips') - al) / al, bootstrap=ci)


def make_report(specification, output):
    spec = load_json(specification)
    review = dict(spec.get('review', {}))
    # Cross-check claims against experiment records instead of trusting report checkboxes alone.
    for run in spec.get('comparisons', []):
        control = load_json(Path(run['control_metrics']).parent / 'summary.json')
        proposed = load_json(Path(run['proposed_metrics']).parent / 'summary.json')
        ga, gb = control['generation'], proposed['generation']
        if ga['config']['experiment'] != 'A1' or gb['config']['experiment'] != 'A3':
            raise ResearchError('Server gate must compare A3 with the matched A1 control.')
        shared_keys = ('render_seed', 'size', 'adapter_width', 'chunk_rows', 'checkpoint_rays', 'amp',
                       'accumulation', 'steps', 'learning_rate', 'weight_decay', 'perceptual_weight')
        if any(ga['config'][k] != gb['config'][k] for k in shared_keys):
            raise ResearchError('Compared experiments use different training or rendering resources.')
        if ga['config']['seed'] != run['seed'] or gb['config']['seed'] != run['seed']:
            raise ResearchError('Comparison seed does not match its trained checkpoints.')
        if any(ga[k] != gb[k] for k in ('manifest_sha256', 'style_sha256', 'stress', 'trained_steps')):
            raise ResearchError('Comparison data, style, perturbation, or optimizer budget differs.')
        if any(ga[k] != gb[k] for k in ('train_manifest_sha256', 'train_data_digest')):
            raise ResearchError('Compared experiments were trained on different data.')
        if not ga.get('audit_freeze_sha256') or ga['audit_freeze_sha256'] != gb.get('audit_freeze_sha256'):
            review['audit_frozen_before_evaluation'] = False
        if any(r['split'] != 'audit' for path in (run['control_metrics'], run['proposed_metrics']) for r in read_jsonl(path)):
            raise ResearchError('Server decisions require final audit metrics, not development results.')
        if control['segmenter_revision'] != proposed['segmenter_revision']:
            raise ResearchError('Evaluation segmenters differ between compared experiments.')
        if min(control['manually_reviewed_count'], proposed['manually_reviewed_count']) == 0:
            review['human_review_complete'] = False
    results = [comparison(run['control_metrics'], run['proposed_metrics'], run['seed']) for run in spec.get('comparisons', [])]
    gate = server_gate(results, review)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'decision.json', dict(gate=gate, comparisons=results))
    with (out / 'comparisons.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed', 'relative_boundary_reduction', 'iou_change', 'relative_lpips_change'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in results)
    text = f"# Laptop-to-server decision\n\nStatus: **{gate['status']}**.\n\n"
    if gate.get('missing'):
        text += 'Missing evidence: ' + ', '.join(gate['missing']) + '.\n'
    text += '\nThis report does not imply that generated facade details are observed facts. Reliability is a separate claim.\n'
    (out / 'report.md').write_text(text, encoding='utf-8')
    return gate


def ensemble_scores(evaluation_dirs, output):
    if len(evaluation_dirs) < 3:
        raise ResearchError('Reliability requires at least three independently trained adapters.')
    summaries = [load_json(Path(directory) / 'summary.json') for directory in evaluation_dirs]
    generations = [s['generation'] for s in summaries]
    if len({g['config']['seed'] for g in generations}) != len(generations):
        raise ResearchError('Ensemble members must have distinct training seeds.')
    if any(g['config']['experiment'] != 'A3' or not g['checkpoint_sha256'] for g in generations):
        raise ResearchError('Reliability currently targets independently trained A3 adapters.')
    for key in ('manifest_sha256', 'style_sha256', 'stress', 'trained_steps'):
        if len({g[key] for g in generations}) != 1:
            raise ResearchError(f'Ensemble members differ in {key}.')
    if len({s['segmenter_revision'] for s in summaries}) != 1:
        raise ResearchError('Ensemble segmenter revisions differ.')
    results = [read_jsonl(Path(directory) / 'metrics.jsonl') for directory in evaluation_dirs]
    maps = [{r['sample_id']: r for r in rows} for rows in results]
    if any(set(m) != set(maps[0]) for m in maps):
        raise ResearchError('Ensemble evaluations have different sample sets.')
    rows = []
    for sid in sorted(maps[0]):
        probabilities = np.stack([np.load(Path(directory) / 'probabilities' / f'{sid}.npy') for directory in evaluation_dirs])
        source = maps[0][sid]
        rows.append(dict(sample_id=sid, split=source['split'], geo_group=source['geo_group'],
                          # Predict mean adapter error, not an unmeasured error of a new ensemble RGB output.
                          boundary_error=float(np.mean([m[sid]['boundary_error'] for m in maps])),
                          disagreement=float(probabilities.var(axis=0, ddof=1).mean()),
                          entropy=float(np.mean([m[sid]['entropy'] for m in maps]))))
    write_jsonl(output, rows)
    return dict(count=len(rows), error_target='mean_structural_error_across_adapters')
