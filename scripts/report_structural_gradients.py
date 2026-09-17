"""Aggregate the unchanged-model, training-only gradient diagnosis."""
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from satground.common import ResearchError, load_json, read_jsonl, require_development, sha256, utc_now


STEPS = (0, 300, 500)


def validate_records(records, expected_views=100):
    selected = {step: [r for r in records if r['checkpoint_step'] == step] for step in STEPS}
    if len(records) != expected_views * len(STEPS):
        raise ResearchError('Incomplete gradient cohort.')
    previous = None
    for rows in selected.values():
        require_development(rows, {'train'})
        if len(rows) != expected_views:
            raise ResearchError('Incomplete checkpoint cohort.')
        current = {r['sample_id']: tuple(r[k] for k in ('geo_group', 'city', 'valid_pixels',
            'interior_valid_pixels', 'scored_building_pixels', 'target_boundary_pixels')) for r in rows}
        if previous is not None and previous != current:
            raise ResearchError('Training cohort or labels changed between model states.')
        previous = current
        for r in rows:
            if r['valid_pixels'] < r['interior_valid_pixels'] or r['target_boundary_pixels'] > r['interior_valid_pixels']:
                raise ResearchError('Invalid boundary support counts.')
            if r['image_gradient_sum_relative_error'] > 5e-5 or r['adapter_gradient_sum_relative_error'] > 2e-4:
                raise ResearchError('Objective decomposition failed its numerical check.')
    return selected


def distribution(rows, getter):
    groups = defaultdict(list)
    values = []
    for r in rows:
        value = getter(r)
        if value is None:
            continue
        if not math.isfinite(value):
            raise ResearchError('Non-finite gradient statistic.')
        values.append(value); groups[r['geo_group']].append(value)
    return dict(available=len(values), unavailable=len(rows)-len(values),
        median=float(np.median(values)) if values else None,
        q25=float(np.quantile(values, .25)) if values else None,
        q75=float(np.quantile(values, .75)) if values else None,
        minimum=min(values) if values else None, maximum=max(values) if values else None,
        equal_group_mean=float(np.mean([np.mean(v) for v in groups.values()])) if values else None)


def summarize(rows):
    result = dict(views=len(rows), geographic_groups=len({r['geo_group'] for r in rows}))
    for space in ('image_gradients', 'adapter_gradients'):
        result[space] = {kind: {key: distribution(rows, lambda r, s=space, k=kind, key=key: r[s][k][key])
            for key in rows[0][space][kind]} for kind in ('weighted_norms', 'cosines', 'boundary_norm_ratios')} if rows else None
    result['support'] = {key: distribution(rows, lambda r, key=key: r[key]) for key in (
        'valid_pixels', 'interior_valid_pixels', 'scored_building_pixels', 'target_boundary_pixels')}
    result['target_boundary_fraction'] = distribution(rows, lambda r:
        r['target_boundary_pixels'] / r['interior_valid_pixels'] if r['interior_valid_pixels'] else None)
    result['raw_losses'] = {key: distribution(rows, lambda r, key=key: r['raw_losses'][key])
                            for key in ('rgb', 'lpips', 'region', 'boundary', 'total')}
    result['negative_boundary_region_cosines'] = sum(
        r['adapter_gradients']['cosines']['boundary_vs_region'] is not None and
        r['adapter_gradients']['cosines']['boundary_vs_region'] < 0 for r in rows)
    result['negative_boundary_A2_cosines'] = sum(
        r['adapter_gradients']['cosines']['boundary_vs_A2'] is not None and
        r['adapter_gradients']['cosines']['boundary_vs_A2'] < 0 for r in rows)
    result['zero_boundary_gradients'] = sum(r['adapter_gradients']['weighted_norms']['boundary'] == 0 for r in rows)
    return result


def report(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ResearchError('Use a fresh aggregate output directory.')
    status = load_json(root / 'status.json')
    if (status['status'] != 'complete' or status['completed'] != 300 or status['expected'] != 300 or
            status['smoke_test'] or status['optimizer_steps'] != 0 or not status['parameters_unchanged'] or
            not status['frozen_models_verified'] or not status['tf32_disabled_for_gradient_decomposition']):
        raise ResearchError('Full, unchanged-parameter probe is required.')
    if any(sha256(p) != h for p, h in status['identity']['files'].items()):
        raise ResearchError('Pinned input or diagnostic source changed.')
    if sha256(root / 'records.jsonl') != status['records_sha256']:
        raise ResearchError('Raw gradient records changed.')
    records = read_jsonl(root / 'records.jsonl')
    selected = validate_records(records)
    original = read_jsonl('data/manifests/learning100/train.jsonl')
    if {r['sample_id'] for r in original} != {r['sample_id'] for r in selected[0]}:
        raise ResearchError('Probe omitted or added a training view.')
    result = dict(created_utc=utc_now(), study='training_only_gradient_and_pairing_diagnosis',
        optimizer_steps=0, parameters_unchanged=True, validation_targets_used_for_gradients=False,
        views_per_state=100, total_checks=300, model_states=list(STEPS), weights=status['weights'],
        tf32_disabled_for_gradient_decomposition=True, diagnostic_seed_policy='fixed_per_training_sample_shared_across_states',
        peak_vram_mib=status['peak_vram_mib'], elapsed_seconds=status['elapsed_seconds'],
        source_identity_sha256=sha256(root / 'status.json'), records_sha256=status['records_sha256'],
        pipeline_source_hash=status['identity']['pipeline_source_hash'],
        max_image_gradient_sum_relative_error=max(r['image_gradient_sum_relative_error'] for r in records),
        max_adapter_gradient_sum_relative_error=max(r['adapter_gradient_sum_relative_error'] for r in records),
        states={str(step): dict(all=summarize(rows),
            nonempty=summarize([r for r in rows if r['scored_building_pixels'] > 0]),
            empty=summarize([r for r in rows if r['scored_building_pixels'] == 0])) for step, rows in selected.items()},
        limitations=['Raw gradients before AdamW, accumulation, clipping and weight decay.',
            'All 100 existing training views; labels remain unverified machine pseudo-labels.',
            'Gradient magnitude and alignment do not prove causal generalization.',
            'Diagnostic FP32 arithmetic differs from default TF32 training math.',
            'Pairing, absolute heading, metric scale and acquisition dates remain unverified.'])
    output.mkdir(parents=True)
    import json
    (output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    lines = ['# Existing objective gradients on training views', '',
        '100 unchanged training views at three adapter states; no optimizer updates. Weighted gradients before AdamW.', '',
        '| A3 state | Boundary / region norm | Boundary / A2 norm | Cosine A2 vs A3 | Boundary vs A2 negative |',
        '|---|---:|---:|---:|---:|']
    for step in STEPS:
        block = result['states'][str(step)]['all']; a = block['adapter_gradients']
        lines.append(f"| {step} | {a['boundary_norm_ratios']['region']['median']:.4f} | "
            f"{a['boundary_norm_ratios']['A2']['median']:.4f} | {a['cosines']['A2_vs_A3']['median']:.6f} | "
            f"{block['negative_boundary_A2_cosines']}/100 |")
    lines += ['', 'Ratios and cosines are per-image medians, not ratios of aggregated losses. A2 here means the A2 objective at the SAME A3 parameters.',
        'Undefined ratios/cosines remain unavailable. Full distributions, equal-group means and target-defined strata are in the summary.',
        '', '[Summary](summary.json). [Interpretation](../interpretation.md).', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    plot(selected, output / 'gradients.png')
    return result


def plot(selected, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in [(axes[0], 'boundary_norm_ratios', 'Boundary norm / A2 norm'),
                            (axes[1], 'cosines', 'Cosine of A2 and A3 objective gradients')]:
        value_key = 'A2' if key == 'boundary_norm_ratios' else 'A2_vs_A3'
        series = [[r['adapter_gradients'][key][value_key] for r in selected[s]
                   if r['adapter_gradients'][key][value_key] is not None] for s in STEPS]
        ax.boxplot(series, tick_labels=[str(s) for s in STEPS], showfliers=True)
        ax.set_title(title, fontsize=11); ax.set_xlabel('Saved A3 adapter step'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Training-only gradient probe | all 100 views, no updates', fontsize=13)
    fig.text(.5, .01, 'Raw weighted gradients before AdamW. Boxes: median and quartiles; whiskers: 1.5 IQR. FP32 diagnostic math.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .93)); fig.savefig(output, dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/structural-gradients100')
    p.add_argument('--output', required=True)
    a = p.parse_args(); report(a.root, a.output)
