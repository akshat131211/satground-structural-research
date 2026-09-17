"""Audit all saved checkpoint evaluations and publish aggregate-only diagnostics."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from audit_exploratory500 import require, verified_predictions
from satground.common import load_json, read_jsonl, sha256, utc_now
from satground.metrics import grouped_mean


def strata(rows):
    """Use target labels only; never select a stratum from model performance."""
    require(bool(rows), 'Empty cohort.')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate evaluation row.')
    result = {}
    for name, selected in (
            ('target_building', [r for r in rows if r['target_building_pixels'] > 0]),
            ('target_empty', [r for r in rows if r['target_building_pixels'] == 0]),
            ('reviewed', [r for r in rows if r['label_provenance'] == 'manually_reviewed'])):
        result[name] = dict(views=len(selected), groups=len({r['geo_group'] for r in selected}),
                            metrics={k: grouped_mean(selected, k) for k in
                                     ('boundary_error', 'building_iou', 'lpips', 'ssim')} if selected else None,
                            maximum_boundary_penalty_views=sum(r['boundary_error'] == 1 for r in selected))
        if name == 'target_empty':
            require(all(r['valid_pixels'] > 0 for r in selected), 'No scored pixels.')
            enriched = [dict(r, false_positive_fraction=r['predicted_building_pixels'] / r['valid_pixels'])
                        for r in selected]
            result[name].update(nonempty_predictions=sum(r['predicted_building_pixels'] > 0 for r in selected),
                predicted_building_pixels=sum(r['predicted_building_pixels'] for r in selected),
                valid_pixels=sum(r['valid_pixels'] for r in selected),
                group_weighted_false_positive_fraction=grouped_mean(enriched, 'false_positive_fraction') if enriched else None)
        elif name == 'target_building':
            result[name]['zero_building_predictions'] = sum(r['predicted_building_pixels'] == 0 for r in selected)
    return result


def verify_evaluation(folder, generation_folder, reference, expected_step, checkpoint=None):
    summary = load_json(folder / 'summary.json')
    rows = read_jsonl(folder / 'metrics.jsonl')
    generation = summary['generation']
    require(len(rows) == 24 and len({r['sample_id'] for r in rows}) == 24, 'Expected 24 unique views.')
    require(all(r['split'] == 'validation' for r in rows), 'Unexpected evaluation split.')
    require(generation['trained_steps'] == expected_step, 'Wrong checkpoint step.')
    require(generation['scientific'] and generation['target_images_used_for_conditioning'] is False,
            'Expected actual inputs without target conditioning.')
    require(load_json(generation_folder / 'generation.json') == generation and
            sha256(generation_folder / 'generation.json') == summary['generation_sha256'], 'Generation record changed.')
    images = verified_predictions(generation_folder)
    require(set(images) == {r['sample_id'] for r in rows}, 'Prediction cohort mismatch.')
    if checkpoint:
        require(sha256(checkpoint) == generation['checkpoint_sha256'], 'Checkpoint hash mismatch.')
    for key in ('manifest_sha256', 'manual_index_sha256', 'segmenter', 'segmenter_revision'):
        require(summary[key] == reference[key], 'Evaluation mismatch: ' + key)
    for key in ('style_sha256', 'manifest_sha256', 'stress', 'train_manifest_sha256'):
        require(generation[key] == reference['generation'][key], 'Generation mismatch: ' + key)
    for key in ('size', 'render_seed', 'chunk_rows', 'checkpoint_rays', 'amp'):
        require(generation['config'][key] == reference['generation']['config'][key], 'Renderer mismatch: ' + key)
    for key in ('pipeline_source_hash', 'code_revision', 'model_revision', 'packages'):
        require(summary['evaluation_provenance'][key] == reference['evaluation_provenance'][key],
                'Evaluation source mismatch: ' + key)
        require(generation['provenance'][key] == reference['generation']['provenance'][key],
                'Generation source mismatch: ' + key)
    for metric, value in summary['group_weighted'].items():
        require(math.isclose(value, grouped_mean(rows, metric), abs_tol=1e-12), 'Aggregate mismatch.')
    return summary, rows


def report(root, output, local_output):
    root, output, local_output = map(Path, (root, output, local_output))
    require(not output.exists() and not local_output.exists(), 'Use fresh report directories.')
    status = load_json(root / 'status.json')
    require(status['status'] == 'complete' and len(status['completed']) == 16, 'Trajectory evaluation incomplete.')
    require(all(sha256(p) == d for p, d in status['identity']['files'].items()), 'Pinned input changed.')
    reference_folder = Path('runs/exploratory500-A0-reference-eval')
    reference = load_json(reference_folder / 'summary.json')
    index_path = Path('data/review/manual-scored-first3/approved-index.json')
    require(sha256(index_path) == reference['manual_index_sha256'], 'Reviewed index changed.')
    index = load_json(index_path)
    runs, raw, paths = [], {}, {}
    labels = None
    for name, step in [('A0', 0)] + [(n, s) for n in ('A1', 'A3') for s in (100, 200, 300, 400, 500)]:
        if name == 'A0':
            folder, generation_folder = reference_folder, Path('runs/exploratory500-A0-reference')
            checkpoint = None
        elif step == 500:
            generation_folder = Path('runs/exploratory500-seed17') / f'{name}-validation'
            folder = Path(str(generation_folder) + '-eval')
            checkpoint = Path('runs/exploratory500-seed17') / f'{name}-train/last.pt'
        else:
            generation_folder = root / f'{name}-{step:06d}'
            folder = Path(str(generation_folder) + '-eval')
            checkpoint = Path('runs/exploratory500-seed17') / f'{name}-train/step-{step:06d}.pt'
        summary, records = verify_evaluation(folder, generation_folder, reference, step, checkpoint)
        require(summary['generation']['config']['experiment'] == name, 'Experiment label mismatch.')
        current_labels = {r['sample_id']: tuple(r[k] for k in
                          ('geo_group', 'label_provenance', 'valid_pixels', 'target_building_pixels')) for r in records}
        if labels is None:
            labels = current_labels
        require(current_labels == labels, 'Target labels or cohort changed.')
        record = dict(model=name, step=step, metrics=summary['group_weighted'], strata=strata(records),
                      evaluation_sha256=sha256(folder / 'summary.json'), metrics_sha256=sha256(folder / 'metrics.jsonl'),
                      checkpoint_sha256=summary['generation']['checkpoint_sha256'])
        if checkpoint:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(state['step'] == step, 'Saved state step mismatch.')
            record['adapter_parameter_l2'] = math.sqrt(sum(float(v.double().square().sum()) for v in state['adapter'].values()))
            record['adapter_tensor_norms'] = {k: float(v.double().norm()) for k, v in state['adapter'].items()}
        runs.append(record)
        raw[name, step] = {r['sample_id']: r for r in records}
        paths[name, step] = generation_folder
    windows = {}
    for name in ('A1', 'A3'):
        log = read_jsonl(Path('runs/exploratory500-seed17') / f'{name}-train/training.jsonl')
        require([r['step'] for r in log] == list(range(1, 501)), 'Incomplete training log.')
        windows[name] = []
        for start in range(0, 500, 100):
            selected = log[start:start + 100]
            windows[name].append(dict(first_step=start + 1, last_step=start + 100,
                means={k: float(np.mean([r[k] for r in selected])) for k in
                       ('rgb', 'lpips', 'gradient_norm', 'total')},
                gradient_clip_fraction=sum(r['gradient_norm'] > 1.0 for r in selected) / 100))
    reviewed = [dict(review_view_number=item['review_view_number'],
                    trajectory=[dict(model=n, step=s, **{k: raw[n, s][sid][k] for k in
                                ('boundary_error', 'building_iou', 'lpips')}) for n, s in raw])
                for sid, item in sorted(index.items(), key=lambda p: p[1]['review_view_number'])]
    result = dict(created_utc=utc_now(), study='post_hoc_development_diagnosis', server_gate='not_eligible',
                  original_final_step_comparison_unchanged=True, new_training_performed=False,
                  steps=[0, 100, 200, 300, 400, 500], validation_views=24,
                  manifest_sha256=reference['manifest_sha256'], manual_index_sha256=reference['manual_index_sha256'],
                  source_identity_sha256=sha256(root / 'status.json'), runs=runs,
                  training_windows=windows, reviewed_views=reviewed)
    output.mkdir(parents=True)
    local_output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n',
                                       encoding='utf-8', newline='\n')
    lines = ['# Checkpoint trajectory: unchanged model and evaluation', '',
             'Post-hoc development diagnosis. All 24 views retained; no new training or primary-checkpoint selection.', '',
             '| Model | Step | All-view boundary | Building-target boundary | Building-target IoU | LPIPS | Empty-target FP area (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        m, b, e = run['metrics'], run['strata']['target_building'], run['strata']['target_empty']
        lines.append(f"| {run['model']} | {run['step']} | {m['boundary_error']:.6f} | "
                     f"{b['metrics']['boundary_error']:.6f} | {b['metrics']['building_iou']:.6f} | "
                     f"{m['lpips']:.6f} | {100 * e['group_weighted_false_positive_fraction']:.6f} |")
    counts = runs[0]['strata']
    lines += ['', f"Target partition: {counts['target_building']['views']} building-containing scored masks, "
              f"{counts['target_empty']['views']} empty scored masks. Three reviewed targets; others are pseudo-labels.",
              'Each stratum uses equal geographic-group weighting within that stratum. Stratum means do not sum to the all-view mean.',
              'FP area is predicted building pixels divided by valid pixels in empty targets, then averaged by geographic group.',
              'Original empty-boundary penalties are unchanged. A nonempty scored mask does not guarantee a nonempty evaluable boundary.',
              '', '[Aggregate measurements and reviewed-view trajectories](summary.json).', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    plot_trajectory(result, output / 'trajectory.png')
    for name in ('A1', 'A3'):
        canvas = Image.new('RGB', (7 * 256, 3 * 308), '#121922')
        draw = ImageDraw.Draw(canvas)
        for rownum, (sid, item) in enumerate(sorted(index.items(), key=lambda p: p[1]['review_view_number'])):
            sequence = [('Real target', Path(item['target_image'])), ('Frozen A0', paths['A0', 0] / (sid + '.png'))]
            sequence += [(f'{name}: step {s}', paths[name, s] / (sid + '.png')) for s in (100, 200, 300, 400, 500)]
            for col, (label, path) in enumerate(sequence):
                canvas.paste(Image.open(path).convert('RGB'), (col * 256, rownum * 308 + 24))
                draw.text((col * 256 + 5, rownum * 308 + 5), label, fill='white')
                if col >= 2:
                    r = raw[name, (col - 1) * 100][sid]
                    draw.text((col * 256 + 5, rownum * 308 + 284),
                              f"Boundary {r['boundary_error']:.4f} | IoU {r['building_iou']:.3f}", fill='white')
                elif col == 0:
                    draw.text((5, rownum * 308 + 284), f"Reviewed view {item['review_view_number']}", fill='white')
        canvas.save(local_output / (name + '-reviewed-trajectory.png'))
    return result


def plot_trajectory(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline = result['runs'][0]
    definitions = [
        ('All-view boundary error (lower better)', lambda r: r['metrics']['boundary_error']),
        ('Building-target IoU (higher better; 22 views)', lambda r: r['strata']['target_building']['metrics']['building_iou']),
        ('All-view LPIPS (lower better)', lambda r: r['metrics']['lpips']),
        ('Empty-target false-positive area (%) (2 views)', lambda r: 100 * r['strata']['target_empty']['group_weighted_false_positive_fraction'])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, measure) in zip(axes.flat, definitions):
        ax.axhline(measure(baseline), color='#737373', linestyle='--', label='Frozen A0')
        for name, color in [('A1', '#d46b24'), ('A3', '#177da1')]:
            records = [r for r in result['runs'] if r['model'] == name]
            ax.plot([0] + [r['step'] for r in records], [measure(baseline)] + [measure(r) for r in records],
                    '-o', color=color, label=name, markersize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Optimizer steps')
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Saved-checkpoint diagnosis | seed 17 | 24 development views', fontsize=14)
    fig.text(.5, .01, 'Group-weighted descriptive means. Original scores retained; no checkpoint selected. Mostly pseudo-labels.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/checkpoint-diagnosis-seed17')
    parser.add_argument('--output', default='reports/checkpoint-diagnosis')
    parser.add_argument('--local-output', default='runs/checkpoint-diagnosis-gallery')
    args = parser.parse_args()
    result = report(args.root, args.output, args.local_output)
    print(json.dumps({'runs_verified': len(result['runs']), 'output': args.output}))
