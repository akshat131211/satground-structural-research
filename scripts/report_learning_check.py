"""Summarize matched preliminary runs without invoking the main-study server gate."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from satground.common import ResearchError, load_json, read_jsonl, save_json, sha256, utc_now
from satground.evaluation import comparison
from satground.metrics import grouped_mean


METRICS = ('building_iou', 'boundary_error', 'lpips', 'ssim')
SHARED_CONFIG = ('seed', 'render_seed', 'size', 'adapter_width', 'chunk_rows',
                 'checkpoint_rays', 'amp', 'accumulation', 'steps', 'learning_rate',
                 'weight_decay', 'perceptual_weight')


def matched_evidence(summaries, records):
    """Refuse mismatched or non-validation data before reporting any comparison."""
    reference = summaries['A1']['generation']
    reference_rows = {r['sample_id']: r['geo_group'] for r in records['A1']}
    for name in ('A0', 'A1', 'A3'):
        summary, rows = summaries[name], records[name]
        gen = summary['generation']
        if not rows or len({r['sample_id'] for r in rows}) != len(rows):
            raise ResearchError('Missing or duplicate metric rows.')
        if any(r['split'] != 'validation' for r in rows):
            raise ResearchError('Learning-check reporting requires validation data only.')
        if {r['sample_id']: r['geo_group'] for r in rows} != reference_rows:
            raise ResearchError('Learning-check sample IDs or geographic groups differ.')
        if summary['count'] != len(rows) or summary['manifest_sha256'] != gen['manifest_sha256']:
            raise ResearchError('Evaluation count or manifest provenance differs.')
        if not gen.get('scientific') or gen.get('target_images_used_for_conditioning') is not False:
            raise ResearchError('Real images and target-free conditioning are required.')
        if gen['config']['experiment'] != name:
            raise ResearchError('Unexpected experiment identity.')
        for key in ('manifest_sha256', 'style_sha256', 'stress', 'train_manifest_sha256'):
            if gen[key] != reference[key]:
                raise ResearchError('Compared generation provenance differs: ' + key)
        for key in SHARED_CONFIG:
            if gen['config'][key] != reference['config'][key]:
                raise ResearchError('Compared resource settings differ: ' + key)
        for key in ('pipeline_source_hash', 'code_revision', 'model_revision', 'packages'):
            if gen['provenance'][key] != reference['provenance'][key]:
                raise ResearchError('Compared source/model/environment differs: ' + key)
        if summary['segmenter_revision'] != summaries['A1']['segmenter_revision']:
            raise ResearchError('Evaluation segmenter revisions differ.')
        if summary['segmenter'] != summaries['A1']['segmenter'] or not summary['training_segmenter_different']:
            raise ResearchError('Independent evaluation segmenter is required.')
        for metric in METRICS:
            if any(not math.isfinite(r[metric]) for r in rows):
                raise ResearchError('Non-finite evaluation metric.')
            if not math.isclose(grouped_mean(rows, metric), summary['group_weighted'][metric], abs_tol=1e-12):
                raise ResearchError('Evaluation aggregate differs from its per-view records.')
        if name != 'A0':
            if gen['trained_steps'] != reference['trained_steps'] or gen['train_data_digest'] != reference['train_data_digest']:
                raise ResearchError('Training budget or RGB data differs.')
            if not 1 <= gen['trained_steps'] <= 100:
                raise ResearchError('This report is limited to the preliminary 100-step check.')


def report(run_root, output):
    root = Path(run_root)
    summaries, records, methods = {}, {}, {}
    for name in ('A0', 'A1', 'A3'):
        evaluation = root / f'learning100-{name}-validation-eval'
        summaries[name] = load_json(evaluation / 'summary.json')
        records[name] = read_jsonl(evaluation / 'metrics.jsonl')
    matched_evidence(summaries, records)
    for name, summary in summaries.items():
        generation = summary['generation']
        evaluation = root / f'learning100-{name}-validation-eval'
        methods[name] = dict(metrics=summary['group_weighted'],
            checkpoint_sha256=generation['checkpoint_sha256'],
            evaluation_summary_sha256=sha256(evaluation / 'summary.json'),
            per_view_metrics_sha256=sha256(evaluation / 'metrics.jsonl'),
            manually_reviewed_count=summary['manually_reviewed_count'],
            saturated_boundary_penalty_views=sum(r['boundary_error'] == 1 for r in records[name]),
            missing_building_mask_views=sum(r['target_building_pixels'] > 0 and r['predicted_building_pixels'] == 0 for r in records[name]),
            extra_building_mask_views=sum(r['target_building_pixels'] == 0 and r['predicted_building_pixels'] > 0 for r in records[name]))
        if name != 'A0':
            training = root / f'learning100-{name}-seed{generation["config"]["seed"]}'
            budget = load_json(training / 'summary.json')
            log = read_jsonl(training / 'training.jsonl')
            if budget['status'] != 'budget_complete' or budget['step'] != generation['trained_steps']:
                raise ResearchError('Training budget is incomplete.')
            if [r['step'] for r in log] != list(range(1, budget['step'] + 1)):
                raise ResearchError('Training log has missing or duplicate optimizer steps.')
            if any(not math.isfinite(v) for row in log for v in row.values() if isinstance(v, (float, int))):
                raise ResearchError('Non-finite training record.')
            methods[name]['training'] = dict(steps=budget['step'], trainable_parameters=budget['trainable_parameters'],
                elapsed_seconds=budget['elapsed_seconds'], peak_vram_mib=max(r['peak_vram_mib'] for r in log),
                log_sha256=sha256(training / 'training.jsonl'), summary_sha256=sha256(training / 'summary.json'))
    seed = summaries['A1']['generation']['config']['seed']
    contrast = comparison(root / 'learning100-A1-validation-eval/metrics.jsonl',
                          root / 'learning100-A3-validation-eval/metrics.jsonl', seed)
    gen = summaries['A1']['generation']
    if sha256(gen['config']['train_manifest']) != gen['train_manifest_sha256']:
        raise ResearchError('The local training manifest changed after generation.')
    result = dict(stage='preliminary_learning_check', updated_utc=utc_now(),
        validation_views=len(records['A1']), geographic_groups=contrast['bootstrap']['geographic_groups'],
        training_views=len(read_jsonl(gen['config']['train_manifest'])), training_seed=seed,
        manifest_sha256=gen['manifest_sha256'], train_manifest_sha256=gen['train_manifest_sha256'],
        style_sha256=gen['style_sha256'], source=gen['provenance'], methods=methods,
        A3_vs_A1=contrast, server_decision='insufficient_evidence', research_improvement_demonstrated=False,
        limitations=['Single seed and 100 optimizer steps; 24 validation views, not the sealed audit.',
                     'Reference building masks are independent-segmenter pseudo-labels, not human annotations.',
                     'The pretrained model may have seen these training-city holdouts.',
                     'The bootstrap is exploratory with only ten geographic groups.',
                     'Training timing excludes initialization; A1 populated caches that A3 reused.'])
    output = Path(output)
    save_json(output / 'summary.json', result)
    lines = ['# Preliminary 100-step learning check', '',
             'This is a real-data feasibility and learning check. It does not establish a research improvement or authorize server scaling.', '',
             f'{result["training_views"]} training views; {result["validation_views"]} fixed validation views in {result["geographic_groups"]} geographic groups; seed {seed}; 256 x 256 outputs; 100 optimizer steps per adapter.', '',
             '| Experiment | Building IoU (higher) | Boundary error (lower) | LPIPS (lower) | SSIM (higher) |',
             '|---|---:|---:|---:|---:|']
    for name, method in methods.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{method["metrics"][key]:.6f}' for key in METRICS) + ' |')
    lo, hi = contrast['boundary_ci95']
    lines += ['', 'Values give each geographic group equal weight. Boundary distance is normalized by the image diagonal. If only one mask has an assessable boundary after ignoring uncertain pixels, the metric assigns error 1. This includes missing/extra building masks and can also affect tiny uncertain fragments.', '',
              f'A3 versus A1: relative boundary reduction **{100 * contrast["relative_boundary_reduction"]:.2f}%**; paired geographic-bootstrap 95% interval for A1 minus A3 boundary error **[{lo:.6f}, {hi:.6f}]**. Positive values favor A3. IoU change: {100 * contrast["iou_change"]:+.2f} percentage points; relative LPIPS change: {100 * contrast["relative_lpips_change"]:+.2f}%.', '',
              '| Training | Optimizer steps | Trainable parameters | Peak allocated VRAM (MiB) | Loop time (s) |',
              '|---|---:|---:|---:|---:|']
    for name in ('A1', 'A3'):
        t = methods[name]['training']
        lines.append(f'| {name} | {t["steps"]} | {t["trainable_parameters"]:,} | {t["peak_vram_mib"]:.1f} | {t["elapsed_seconds"]:.1f} |')
    counts = ', '.join(f'{name}: {method["saturated_boundary_penalty_views"]}' for name, method in methods.items())
    lines += ['', 'Both logs contain every optimizer step with finite losses and gradient norms. A1 was deliberately stopped after step 10 and resumed to step 100. Cache and initialization differences prevent a controlled speed comparison.', '',
              f'Views receiving the saturated boundary penalty ({counts}) should be checked against human masks. They are retained in all reported primary metrics; no difficult views were removed.', '', '## Limits and next decision', '']
    lines += ['- ' + item for item in result['limitations']]
    lines += ['', 'Complete image alignment, footprint, duplicate, and human boundary review before the main A1/A2/A3 study. Retain matched budgets and three seeds. Seattle and the final laptop audit have not been evaluated.', '',
              'The summary includes source, checkpoint, manifest, training-log, and metric-file hashes. Images and per-location records remain in local ignored directories.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def gallery(run_root, data_root, output):
    """First six preselected validation rows; no score-based example selection."""
    from PIL import Image, ImageDraw
    from satground.common import safe_data_path
    root = Path(run_root)
    gen = load_json(root / 'learning100-A1-validation/generation.json')
    manifest = gen['config']['validation_manifest']
    if sha256(manifest) != gen['manifest_sha256']:
        raise ResearchError('Gallery manifest changed after generation.')
    rows = read_jsonl(manifest)[:6]
    canvas = Image.new('RGB', (1280, 30 + len(rows) * 282), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(('Overhead input', 'Real target', 'A0 frozen', 'A1 RGB + LPIPS', 'A3 structural')):
        draw.text((column * 256 + 6, 8), label, fill='black')
    for number, row in enumerate(rows):
        sid = row['sample_id']
        paths = [safe_data_path(data_root, row['satellite']),
                 root / 'learning100-A0-validation-eval/targets' / f'{sid}.png']
        paths += [root / f'learning100-{name}-validation' / f'{sid}.png' for name in ('A0', 'A1', 'A3')]
        y = 30 + number * 282
        for column, path in enumerate(paths):
            with Image.open(path) as im:
                canvas.paste(im.convert('RGB').resize((256, 256)), (column * 256, y))
        draw.text((6, y + 260), f'{number + 1}. {row["city"]} | {sid} | fixed validation order, 100 steps, seed 17', fill='black')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default='runs')
    parser.add_argument('--output', default='reports/learning100')
    parser.add_argument('--gallery', help='Optional local-only comparison image path; never commit VIGOR imagery')
    parser.add_argument('--data-root', default='data/vigor')
    args = parser.parse_args()
    result = report(args.run_root, args.output)
    if args.gallery:
        gallery(args.run_root, args.data_root, args.gallery)
    print(result['A3_vs_A1'])
