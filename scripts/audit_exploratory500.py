"""Reproduce the post-run diagnostics; retain every primary evaluation view."""
import argparse
import json
import math
from pathlib import Path

import torch

from report_exploratory_pair import report
from satground.common import ResearchError, load_json, read_jsonl, safe_data_path, sha256, utc_now
from satground.evaluation import comparison
from satground.metrics import grouped_mean


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def finite_values(value):
    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all()), 'Non-finite checkpoint tensor.')
        return value.numel()
    if isinstance(value, dict):
        return sum(finite_values(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_values(v) for v in value)
    return 0


def verified_predictions(folder):
    records = read_jsonl(folder / 'predictions.jsonl')
    result = {}
    for row in records:
        require(row['sample_id'] not in result, 'Duplicate prediction.')
        require(sha256(safe_data_path(folder, row['prediction'])) == row['sha256'],
                'Prediction file changed.')
        result[row['sample_id']] = row['sha256']
    return result


def audit(output, work):
    output, work = Path(output), Path(work)
    require(not output.exists() and not work.exists(), 'Use fresh audit output paths.')
    root = Path('runs/exploratory500-seed17')
    paired = report(root, work / 'recomputed-pair')
    published = load_json('reports/exploratory500/summary.json')
    require({k: v for k, v in paired.items() if k != 'created_utc'} ==
            {k: v for k, v in published.items() if k != 'created_utc'},
            'Published paired report differs from recomputation.')
    rows, summaries = {}, {}
    folders = {'A0': Path('runs/exploratory500-A0-reference-eval'),
               **{n: root / f'{n}-validation-eval' for n in ('A1', 'A3')}}
    for name, folder in folders.items():
        records = read_jsonl(folder / 'metrics.jsonl')
        rows[name] = {r['sample_id']: r for r in records}
        require(len(rows[name]) == len(records) == 24, 'Expected 24 unique evaluation views.')
        summaries[name] = load_json(folder / 'summary.json')
        require(all(r['split'] == 'validation' for r in records), 'Unexpected split.')
        for metric, value in summaries[name]['group_weighted'].items():
            require(math.isclose(value, grouped_mean(records, metric), abs_tol=1e-12),
                    'Summary differs from per-view metrics.')
    a0, a1 = summaries['A0'], summaries['A1']
    for key in ('manifest_sha256', 'manual_index_sha256', 'segmenter', 'segmenter_revision'):
        require(a0[key] == a1[key], 'Frozen-reference evaluation mismatch: ' + key)
    for key in ('pipeline_source_hash', 'code_revision', 'model_revision', 'packages'):
        require(a0['evaluation_provenance'][key] == a1['evaluation_provenance'][key],
                'Frozen-reference provenance mismatch: ' + key)
        require(a0['generation']['provenance'][key] == a1['generation']['provenance'][key],
                'Frozen-reference generation mismatch: ' + key)
    for key in ('size', 'render_seed', 'chunk_rows', 'checkpoint_rays', 'amp'):
        require(a0['generation']['config'][key] == a1['generation']['config'][key],
                'Frozen-reference renderer mismatch: ' + key)
    for key in ('manifest_sha256', 'style_sha256', 'train_manifest_sha256', 'stress'):
        require(a0['generation'][key] == a1['generation'][key], 'Frozen-reference input mismatch: ' + key)
    require(a0['generation']['trained_steps'] == 0 and
            a0['generation']['target_images_used_for_conditioning'] is False,
            'Expected frozen A0 without target conditioning.')
    a0_folder = Path('runs/exploratory500-A0-reference')
    require(sha256(a0_folder / 'generation.json') == a0['generation_sha256'] and
            load_json(a0_folder / 'generation.json') == a0['generation'], 'A0 generation record changed.')
    current = verified_predictions(a0_folder)
    old = verified_predictions(Path('runs/learning100-A0-validation'))
    require(current == old and set(current) == set(rows['A0']), 'Frozen A0 images differ.')
    for name in ('A0', 'A3'):
        require({k: (r['geo_group'], r['label_provenance'], r['valid_pixels'], r['target_building_pixels'])
                 for k, r in rows[name].items()} ==
                {k: (r['geo_group'], r['label_provenance'], r['valid_pixels'], r['target_building_pixels'])
                 for k, r in rows['A1'].items()}, 'Evaluation cohort or scoring labels differ.')
    finite = {}
    for name in ('A1', 'A3'):
        checkpoint = torch.load(root / f'{name}-train/last.pt', map_location='cpu', weights_only=True)
        require(checkpoint['step'] == 500, 'Unexpected checkpoint step.')
        finite[name] = {key + '_tensor_values': finite_values(checkpoint[key])
                        for key in ('adapter', 'optimizer')}
    first = {n: read_jsonl(root / f'{n}-train/training.jsonl')[0] for n in ('A1', 'A3')}
    # Compare only the unweighted reconstruction components, before the first update.
    component_keys = [k for k in first['A1'] if 'rgb' in k or 'lpips' in k]
    require(len(component_keys) >= 2 and all(first['A1'][k] == first['A3'][k] for k in component_keys),
            'Initial RGB/perceptual components differ or are missing.')
    index_path = Path('data/review/manual-scored-first3/approved-index.json')
    require(sha256(index_path) == a1['manual_index_sha256'], 'Reviewed index changed.')
    index = load_json(index_path)
    require(len(index) == 3, 'Expected original three reviewed views.')
    reviewed = []
    for sid, item in sorted(index.items(), key=lambda pair: pair[1]['review_view_number']):
        require(all(rows[n][sid]['label_provenance'] == 'manually_reviewed' for n in rows),
                'Expected reviewed labels.')
        reviewed.append({'review_view_number': item['review_view_number'],
                         'metrics': {n: {m: rows[n][sid][m] for m in
                                         ('boundary_error', 'building_iou', 'lpips', 'ssim')} for n in rows}})
    groups = sorted({r['geo_group'] for r in rows['A1'].values()})
    differences = {}
    for group in groups:
        ids = [sid for sid, r in rows['A1'].items() if r['geo_group'] == group]
        differences[group] = sum(rows['A1'][sid]['boundary_error'] - rows['A3'][sid]['boundary_error']
                                 for sid in ids) / len(ids)
    largest = max(differences, key=differences.get)
    dominant = [sid for sid, r in rows['A1'].items() if r['geo_group'] == largest]
    require(len(dominant) == 1, 'Revisit diagnostic narrative: largest group is not one view.')
    sid = dominant[0]
    result = {'verified_utc': utc_now(), 'primary_report_recomputed_equal': True,
              'primary_report_sha256': sha256('reports/exploratory500/summary.json'),
              'all_views_retained': True, 'server_gate': 'not_eligible',
              'finite_checkpoint_tensors': finite, 'initial_rgb_and_lpips_equal': True,
              'a0_reference': {'purpose': 'Post-run check against the frozen model, using the same current labels.',
                               'metrics': a0['group_weighted'], 'unchanged_original_predictions': len(current),
                               'evaluation_sha256': sha256(folders['A0'] / 'summary.json'),
                               'comparison_a3_vs_a0': comparison(folders['A0'] / 'metrics.jsonl',
                                                               folders['A3'] / 'metrics.jsonl', 17)},
              'reviewed_views': reviewed,
              'boundary_sensitivity': {
                  'selection': 'Post-hoc group with largest contribution; retained in primary results.',
                  'group_differences_sorted': sorted(differences.values()),
                  'dominant_group_views': len(dominant),
                  'share_of_net_improvement': differences[largest] / sum(differences.values()),
                  'dominant_view_metrics': {n: {m: rows[n][sid][m] for m in
                      ('boundary_error', 'building_iou', 'target_building_pixels',
                       'predicted_building_pixels', 'valid_pixels')} for n in rows}},
              'interpretation': 'Aggregate A3/A1 gain does not establish structural improvement; '
                                'the empty-mask convention dominates and all three reviewed boundaries worsen.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work', required=True, help='Fresh ignored directory for verification artifacts.')
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.work), indent=2))
