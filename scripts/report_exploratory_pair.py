"""Report a fixed-budget A1/A3 development experiment, never a server decision."""
import argparse
import math
from pathlib import Path

from satground.common import ResearchError, load_json, read_jsonl, save_json, sha256, utc_now
from satground.evaluation import comparison
from satground.metrics import grouped_mean


def report(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ResearchError('Choose a fresh report directory.')
    summaries, rows, methods = {}, {}, {}
    for name in ('A1', 'A3'):
        folder = root / f'{name}-validation-eval'
        summaries[name] = load_json(folder / 'summary.json')
        rows[name] = read_jsonl(folder / 'metrics.jsonl')
    a, b = (summaries[name]['generation'] for name in ('A1', 'A3'))
    allowed_differences = {'experiment', 'region_weight', 'boundary_weight'}
    if {k: v for k, v in a['config'].items() if k not in allowed_differences} != {
            k: v for k, v in b['config'].items() if k not in allowed_differences}:
        raise ResearchError('Matched resources/configuration differ.')
    for key in ('manifest_sha256', 'style_sha256', 'stress', 'trained_steps',
                'train_manifest_sha256', 'train_data_digest', 'training_readiness'):
        if a[key] != b[key]:
            raise ResearchError('Matched generation evidence differs: ' + key)
    for key in ('segmenter', 'segmenter_revision', 'manual_index_sha256'):
        if summaries['A1'][key] != summaries['A3'][key]:
            raise ResearchError('Evaluation labels or segmenter differ.')
    sources = []
    reference_ids = {r['sample_id']: (r['geo_group'], r['label_provenance'], r['valid_pixels']) for r in rows['A1']}
    for name in ('A1', 'A3'):
        s, records = summaries[name], rows[name]
        g = s['generation']
        train = load_json(root / f'{name}-train' / 'summary.json')
        log = read_jsonl(root / f'{name}-train' / 'training.jsonl')
        steps = g['config']['steps']
        if g['config']['experiment'] != name or g['config'].get('data_review_mode') != 'exploratory':
            raise ResearchError('Expected explicitly exploratory A1/A3 experiments.')
        if g['trained_steps'] != steps or train['step'] != steps or train['status'] != 'budget_complete':
            raise ResearchError('Both fixed training budgets must be complete.')
        if [r['step'] for r in log] != list(range(1, steps + 1)) or any(
                not math.isfinite(v) for r in log for v in r.values() if isinstance(v, (int, float))):
            raise ResearchError('Missing steps or non-finite training log.')
        if train['checkpoint_sha256'] != g['checkpoint_sha256'] or sha256(root / f'{name}-train/last.pt') != g['checkpoint_sha256']:
            raise ResearchError('Training checkpoint does not match evaluated predictions.')
        generation_path = root / f'{name}-validation/generation.json'
        if sha256(generation_path) != s['generation_sha256'] or load_json(generation_path) != g:
            raise ResearchError('Generation record changed after evaluation.')
        predicted = read_jsonl(root / f'{name}-validation/predictions.jsonl')
        if len(predicted) != len(records) or {r['sample_id'] for r in predicted} != set(reference_ids):
            raise ResearchError('Prediction cohort differs from evaluated cohort.')
        for p in predicted:
            from satground.common import safe_data_path
            if sha256(safe_data_path(root / f'{name}-validation', p['prediction'])) != p['sha256']:
                raise ResearchError('A generated image changed after generation.')
        if not records or len({r['sample_id'] for r in records}) != len(records) or any(r['split'] != 'validation' for r in records):
            raise ResearchError('Only complete, unique validation cohorts are permitted.')
        if {r['sample_id']: (r['geo_group'], r['label_provenance'], r['valid_pixels']) for r in records} != reference_ids:
            raise ResearchError('Evaluation views, groups, or label coverage differ.')
        if not g['scientific'] or g['target_images_used_for_conditioning'] is not False:
            raise ResearchError('Real inputs without target conditioning are required.')
        for metric, value in s['group_weighted'].items():
            if not math.isclose(value, grouped_mean(records, metric), abs_tol=1e-12):
                raise ResearchError('Saved aggregate differs from per-view metrics.')
        for p in (g['provenance'], g['training_provenance'], s['evaluation_provenance']):
            sources.append(tuple(p[k] if k != 'packages' else tuple(sorted(p[k].items()))
                                 for k in ('pipeline_source_hash', 'code_revision', 'model_revision', 'packages')))
        methods[name] = dict(metrics=s['group_weighted'], checkpoint_sha256=g['checkpoint_sha256'],
                            elapsed_seconds=train['elapsed_seconds'], peak_vram_mib=train['peak_vram_mib'],
                            trainable_parameters=train['trainable_parameters'],
                            evaluation_sha256=sha256(root / f'{name}-validation-eval/summary.json'),
                            training_log_sha256=sha256(root / f'{name}-train/training.jsonl'),
                            manually_reviewed_views=s['manually_reviewed_count'])
    if len(set(sources)) != 1:
        raise ResearchError('Training, generation, or evaluation code/environment differs.')
    delta = comparison(root / 'A1-validation-eval/metrics.jsonl', root / 'A3-validation-eval/metrics.jsonl', a['config']['seed'])
    result = dict(created_utc=utc_now(), study='exploratory_pending_data_review', server_gate='not_eligible',
                  steps_per_model=a['trained_steps'], seed=a['config']['seed'],
                  validation_views=len(rows['A1']), geographic_groups=len({r['geo_group'] for r in rows['A1']}),
                  manifest_sha256=a['manifest_sha256'], manual_index_sha256=summaries['A1']['manual_index_sha256'],
                  readiness=a['training_readiness'], methods=methods, comparison=delta,
                  limitations=['Single seed, previously inspected development images, mostly pseudo-labels.',
                               'Geographic alignment, footprint scale and near-duplicate review remain incomplete.',
                               'Validation bootstrap does not establish audited generalization or publication novelty.'])
    save_json(output / 'summary.json', result)
    lo, hi = delta['boundary_ci95']
    lines = ['# Exploratory 500-step laptop comparison', '',
             f"Each adapter trained {a['trained_steps']} steps from zero initialization, seed {a['config']['seed']}. "
             f"All {len(rows['A1'])} pre-existing validation views are retained; three use reviewed target masks.", '',
             '**This is an exploratory development result with pending data review. It cannot pass the server gate.**', '',
             '| Model | Boundary error | Building IoU | LPIPS |', '|---|---:|---:|---:|']
    for name in ('A1', 'A3'):
        m = methods[name]['metrics']
        lines.append(f"| {name} | {m['boundary_error']:.6f} | {m['building_iou']:.6f} | {m['lpips']:.6f} |")
    lines += ['', f"A3 versus A1: boundary reduction {delta['relative_boundary_reduction']:.2%}; "
              f"paired geographic-bootstrap absolute-improvement 95% CI [{lo:.6f}, {hi:.6f}]. "
              f"IoU change {100 * delta['iou_change']:+.3f} percentage points; LPIPS change {delta['relative_lpips_change']:+.2%}.", '',
              'The final budget was fixed before training; no best-of-checkpoint selection was performed. '
              'Earlier 100-step experiments and their outputs are preserved. Label changes prevent direct '
              'comparison of their published structural aggregates with this mixed-label evaluation.', '',
              *result['limitations'], '', '[Aggregate evidence](summary.json). No images, coordinates or sample identifiers are published.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(result['comparison'])
