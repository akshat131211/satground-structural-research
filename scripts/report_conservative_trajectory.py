"""Audit every conservative checkpoint before considering a longer budget."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from audit_exploratory500 import finite_values, require, verified_predictions
from report_checkpoint_diagnosis import strata, verify_evaluation
from report_reviewed21 import delta
from satground.common import load_json, read_jsonl, save_json, sha256, utc_now
from satground.metrics import grouped_mean


NAMES = ('A1', 'A2', 'A3')
STEPS = (100, 200, 300, 400, 500)


def verify_final_reproduction(original_predictions, regenerated_predictions, original_rows, regenerated_rows):
    require(original_predictions == regenerated_predictions, 'Final prediction images differ.')
    a, b = ({r['sample_id']: r for r in records} for records in (original_rows, regenerated_rows))
    require(len(a) == len(original_rows) and len(b) == len(regenerated_rows) and a == b,
            'Final evaluation differs from preserved expanded-label result.')


def reviewed(rows):
    return [r for r in rows if r['label_provenance'] == 'manually_reviewed']


def late_changes(raw):
    return {name: {str(start) + '_to_500': dict(
        all24=delta(raw[name, start], raw[name, 500]),
        reviewed21=delta(reviewed(raw[name, start]), reviewed(raw[name, 500])))
        for start in (300, 400)} for name in NAMES}


def report(root, output, local_output):
    root, output, local_output = map(Path, (root, output, local_output))
    require(not output.exists() and not local_output.exists(), 'Use fresh output directories.')
    state = load_json(root / 'status.json')
    expected = [f'{n}-{s:06d}-{action}' for s in STEPS for n in NAMES for action in ('generate', 'evaluate')]
    require(state['status'] == 'complete' and state['completed'] == expected, 'Trajectory is incomplete.')
    require(state['new_training_performed'] is False, 'Expected evaluation only.')
    require(all(sha256(p) == h for p, h in state['identity']['files'].items()), 'Pinned inputs changed.')
    reference = load_json(root / 'A1-000100-eval/summary.json')
    index_path = Path('data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json')
    index = load_json(index_path)
    require(len(index) == 21 and sha256(index_path) == reference['manual_index_sha256'], 'Reviewed index differs.')
    train_root = Path('runs/conservative500-seed17')
    records, raw, windows = [], {}, {}
    labels = None
    for name in NAMES:
        log = read_jsonl(train_root / f'{name}-train/training.jsonl')
        require([r['step'] for r in log] == list(range(1, 501)), 'Training log incomplete.')
        require(all(math.isfinite(v) for r in log for v in r.values() if isinstance(v, (int, float))),
                'Training log has non-finite values.')
        windows[name] = []
        for start in range(0, 500, 100):
            selected = log[start:start + 100]
            windows[name].append(dict(first_step=start + 1, last_step=start + 100,
                means={k: float(np.mean([r[k] for r in selected])) for k in selected[0]
                       if k not in ('step', 'elapsed_seconds', 'peak_vram_mib')}))
        for step in STEPS:
            folder = root / f'{name}-{step:06d}'
            evaluation = root / f'{name}-{step:06d}-eval'
            checkpoint = train_root / f'{name}-train/step-{step:06d}.pt'
            summary, rows = verify_evaluation(evaluation, folder, reference, step, checkpoint)
            generation = summary['generation']
            require(generation['config']['experiment'] == name and generation['config']['learning_rate'] == 3e-5,
                    'Not the matched conservative experiment.')
            require(generation['config'] == load_json(train_root / f'{name}-validation/generation.json')['config'],
                    'Configuration differs from original experiment.')
            require(summary['evaluation_provenance']['pipeline_source_hash'] == state['identity']['pipeline_source_hash'],
                    'Evaluation source changed.')
            require(summary['manually_reviewed_count'] == 21 and set(index).issubset({r['sample_id'] for r in rows}),
                    'Reviewed cohort mismatch.')
            current_labels = {r['sample_id']: tuple(r[k] for k in
                ('geo_group', 'target_building_pixels', 'valid_pixels', 'label_provenance')) for r in rows}
            if labels is None:
                labels = current_labels
            require(labels == current_labels, 'Scoring labels changed across checkpoints.')
            saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
            require(saved['step'] == step and finite_values(saved['adapter']) > 0 and finite_values(saved['optimizer']) > 0,
                    'Non-finite or missing checkpoint tensors.')
            if step == 500:
                verify_final_reproduction(verified_predictions(train_root / f'{name}-validation'),
                    verified_predictions(folder),
                    read_jsonl(Path('runs/conservative500-reviewed21') / f'{name}-eval/metrics.jsonl'), rows)
            records.append(dict(model=name, step=step, metrics=summary['group_weighted'], strata=strata(rows),
                checkpoint_sha256=sha256(checkpoint), metrics_sha256=sha256(evaluation / 'metrics.jsonl'),
                evaluation_sha256=sha256(evaluation / 'summary.json')))
            raw[name, step] = rows
    a0_folder = Path('runs/conservative500-reviewed21/A0-eval')
    a0 = load_json(a0_folder / 'summary.json')
    require(a0['manual_index_sha256'] == sha256(index_path) and a0['manifest_sha256'] == reference['manifest_sha256'],
            'A0 reference labels or cohort differ.')
    a0_rows = read_jsonl(a0_folder / 'metrics.jsonl')
    require(verified_predictions(Path('runs/exploratory500-A0-reference')).keys() == labels.keys(),
            'A0 prediction cohort differs.')
    require(all(math.isclose(grouped_mean(a0_rows, k), v, abs_tol=1e-12)
                for k, v in a0['group_weighted'].items()), 'A0 aggregate mismatch.')
    comparisons = {str(step): {a + '_vs_' + b: dict(all24=delta(raw[b, step], raw[a, step]),
                    reviewed21=delta(reviewed(raw[b, step]), reviewed(raw[a, step])))
                    for a, b in (('A3', 'A1'), ('A3', 'A2'), ('A2', 'A1'))} for step in STEPS}
    result = dict(created_utc=utc_now(), study='post_hoc_conservative_checkpoint_diagnosis',
        new_training_performed=False, primary_checkpoint_unchanged=True, no_views_removed=True,
        validated_checkpoints=15, verified_predictions=360, final_prediction_reproductions=72,
        checkpoint_tensors_finite=True, training_log_rows_finite=1500, training_views=100,
        image_presentations_per_model=500 * 8, validation_views=24, reviewed_views=21, geographic_groups=10,
        remaining_pseudo_labels=3, manual_index_sha256=sha256(index_path),
        source_identity_sha256=sha256(root / 'status.json'), runs=records, training_windows=windows,
        frozen_reference=dict(metrics=a0['group_weighted'], strata=strata(a0_rows),
                              evaluation_sha256=sha256(a0_folder / 'summary.json')),
        late_changes=late_changes(raw), matched_comparisons=comparisons,
        limitations=['One training seed and previously inspected development views.',
            'Training losses and late checkpoint comparisons are diagnostic, not audited generalization.',
            'Human-reviewed machine proposals, incomplete pairing/coverage QA, three pseudo-labelled targets.',
            'Unadjusted paired bootstrap intervals; all model states and views retained.'])
    output.mkdir(parents=True)
    save_json(output / 'summary.json', result)
    lines = ['# Does longer training have supporting evidence?', '',
        'All 15 saved model states checked; no new training. Same 24 views and 21 reviewed target-mask pairs.', '',
        '| Model | Step | All boundary | Reviewed boundary | Reviewed IoU | LPIPS | Empty-target FP pixels |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m, s = r['metrics'], r['strata']
        lines.append(f"| {r['model']} | {r['step']} | {m['boundary_error']:.6f} | "
                     f"{s['reviewed']['metrics']['boundary_error']:.6f} | {s['reviewed']['metrics']['building_iou']:.6f} | "
                     f"{m['lpips']:.6f} | {s['target_empty']['predicted_building_pixels']} |")
    lines += ['', 'Means weight geographic groups equally. False-positive pixels are pooled counts on three empty targets.',
              'Lower boundary and LPIPS are better; higher IoU is better. Earlier reports and final checkpoint choice are unchanged.',
              '', '[Interpretation](interpretation.md). [All paired intervals and strata](summary.json).', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    plot(result, output / 'trajectory.png')
    local_output.mkdir(parents=True)
    original = load_json('data/review/manual-scored-first3/approved-index.json')
    for name in NAMES:
        canvas = Image.new('RGB', (7 * 192, 3 * 214), '#17202a')
        draw = ImageDraw.Draw(canvas)
        for i, (sid, entry) in enumerate(sorted(original.items(), key=lambda pair: pair[1]['review_view_number'])):
            views = [('Target', Path(entry['target_image'])), ('A0', Path('runs/exploratory500-A0-reference') / (sid + '.png'))]
            views += [(f'{name}: {step}', root / f'{name}-{step:06d}' / (sid + '.png')) for step in STEPS]
            for j, (label, path) in enumerate(views):
                with Image.open(path) as image:
                    canvas.paste(image.convert('RGB').resize((192, 192)), (j * 192, i * 214 + 22))
                draw.text((j * 192 + 4, i * 214 + 4), label, fill='white')
        canvas.save(local_output / (name + '-original-three.png'))
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    definitions = [
        ('Reviewed boundary error (21 views)', lambda r: r['strata']['reviewed']['metrics']['boundary_error']),
        ('Building-target boundary error (21 views)', lambda r: r['strata']['target_building']['metrics']['boundary_error']),
        ('LPIPS (24 views)', lambda r: r['metrics']['lpips']),
        ('Empty-target false-positive area (%)', lambda r: 100 * r['strata']['target_empty']['group_weighted_false_positive_fraction'])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, measure) in zip(axes.flat, definitions):
        ax.axhline(measure(result['frozen_reference']), color='#777777', linestyle='--', label='Frozen A0')
        for name, color in zip(NAMES, ('#d46b24', '#548448', '#177da1')):
            rows = [r for r in result['runs'] if r['model'] == name]
            ax.plot([r['step'] for r in rows], [measure(r) for r in rows], '-o', color=color, label=name, markersize=4)
        ax.set_title(title + ' | lower better', fontsize=10)
        ax.set_xlabel('Optimizer steps'); ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Conservative schedule: saved checkpoints, unchanged labels', fontsize=13)
    fig.text(.5, .01, 'Exploratory, seed 17. Equal geographic-group means; uncertainty in accompanying tables. No checkpoint selected.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95)); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/conservative-trajectory-reviewed21')
    p.add_argument('--output', required=True)
    p.add_argument('--local-output', required=True)
    a = p.parse_args()
    r = report(a.root, a.output, a.local_output)
    print('Verified', r['validated_checkpoints'], 'checkpoints and', r['verified_predictions'], 'predictions.')
