"""Recompute the completed matched revision and retain aggregate-only evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_exploratory500 import require, verified_predictions
from report_checkpoint_diagnosis import strata, verify_evaluation
from report_conservative_adaptation import report
from satground.annotations import verified_reviewed_masks
from satground.common import load_json, provenance, read_jsonl, sha256, utc_now
from satground.evaluation import comparison


def without_time(record):
    return {k: v for k, v in record.items() if k != 'created_utc'}


def audit(output, work):
    output, work = Path(output), Path(work)
    require(not output.exists() and not work.exists(), 'Use fresh audit paths.')
    root = Path('runs/conservative500-seed17')
    state = load_json(root / 'status.json')
    expected = [n + suffix for n in ('A1', 'A2', 'A3')
                for suffix in ('-train', '-validation', '-validation-eval')]
    require(state['status'] == 'complete' and state['completed'] == expected,
            'All nine stages must finish.')
    require(all(sha256(p) == h for p, h in state['identity']['files'].items()),
            'Pinned inputs changed.')
    require(provenance()['pipeline_source_hash'] == state['identity']['pipeline_source_hash'],
            'Pipeline source changed.')
    stage_logs = {}
    for stage in expected:
        log_path = root / (stage + '.log')
        text = log_path.read_text(encoding='utf-8')
        require(text and 'Traceback (most recent call last)' not in text and
                'CUDA out of memory' not in text, 'Inspect stage failure: ' + stage)
        stage_logs[stage] = sha256(log_path)
    recomputed = report(root, work / 'recomputed')
    published = Path('reports/conservative500')
    require(without_time(recomputed) == without_time(load_json(published / 'summary.json')),
            'Ablation report differs from recomputation.')
    require(without_time(load_json(work / 'recomputed/primary/summary.json')) ==
            without_time(load_json(published / 'primary/summary.json')),
            'Primary paired report differs from recomputation.')
    reference = load_json(root / 'A1-validation-eval/summary.json')
    folders = {'A0': Path('runs/exploratory500-A0-reference-eval'),
               **{n: root / f'{n}-validation-eval' for n in ('A1', 'A2', 'A3')}}
    predictions = {'A0': Path('runs/exploratory500-A0-reference'),
                   **{n: root / f'{n}-validation' for n in ('A1', 'A2', 'A3')}}
    raw, summaries, labels = {}, {}, None
    for name, folder in folders.items():
        checkpoint = None if name == 'A0' else root / f'{name}-train/last.pt'
        summary, rows = verify_evaluation(folder, predictions[name], reference,
                                          0 if name == 'A0' else 500, checkpoint)
        current = {r['sample_id']: tuple(r[k] for k in
                   ('geo_group', 'label_provenance', 'valid_pixels', 'target_building_pixels')) for r in rows}
        if labels is None:
            labels = current
        require(current == labels, 'A0/adaptation labels differ.')
        raw[name] = {r['sample_id']: r for r in rows}
        summaries[name] = summary
    require(verified_predictions(predictions['A0']) ==
            verified_predictions(Path('runs/learning100-A0-validation')), 'Original A0 images changed.')
    index_path = Path('data/review/manual-scored-first3/approved-index.json')
    require(sha256(index_path) == reference['manual_index_sha256'], 'Fixed reviewed index changed.')
    index = load_json(index_path)
    require(len(index) == 3, 'Expected exactly three original reviewed targets.')
    reviewed = []
    canvas = Image.new('RGB', (5 * 256, 3 * 312), '#121922')
    draw = ImageDraw.Draw(canvas)
    for rownum, (sid, entry) in enumerate(sorted(index.items(), key=lambda p: p[1]['review_view_number'])):
        target = np.asarray(Image.open(entry['target_image']).convert('RGB'))
        verified_reviewed_masks(entry, target)
        require(all(raw[n][sid]['label_provenance'] == 'manually_reviewed' for n in raw),
                'Reviewed target incorrectly labelled.')
        reviewed.append(dict(review_view_number=entry['review_view_number'], metrics={
            n: {k: raw[n][sid][k] for k in ('boundary_error', 'building_iou', 'lpips', 'ssim')}
            for n in raw}))
        sequence = [('Real target', Path(entry['target_image']))] + [
            (n, predictions[n] / (sid + '.png')) for n in raw]
        for col, (label, path) in enumerate(sequence):
            with Image.open(path) as image:
                require(image.size == (256, 256), 'Unexpected gallery image dimensions.')
                canvas.paste(image.convert('RGB'), (col * 256, rownum * 312 + 24))
            draw.text((col * 256 + 5, rownum * 312 + 5), label, fill='white')
            caption = f"Reviewed view {entry['review_view_number']}" if col == 0 else (
                f"Boundary {raw[label][sid]['boundary_error']:.4f} | IoU {raw[label][sid]['building_iou']:.3f}")
            draw.text((col * 256 + 5, rownum * 312 + 284), caption, fill='white')
    gallery = work / 'reviewed-three-view-comparison.png'
    canvas.save(gallery)
    first = {n: read_jsonl(root / f'{n}-train/training.jsonl')[0] for n in ('A1', 'A2', 'A3')}
    require(all(first[n][k] == first['A1'][k] for n in first for k in ('rgb', 'lpips')),
            'Initial reconstruction/perceptual losses differ.')
    review = load_json('reports/ADDITIONAL_REVIEW_STATUS.json')
    count = review['reviewed_count']
    approved_root = Path('data/review/checkpoint-review18-approved')
    progress = load_json(approved_root / f'progress-v{count}.json')
    require(progress['reviewed_count'] == count and len(progress['accepted_views']) == count,
            'Additional review count mismatch.')
    require(sha256(progress['batch_index']) == progress['batch_index_sha256'] ==
            review['approved_batch_index_sha256'], 'Additional approved index changed.')
    for row in progress['accepted_views']:
        require(sha256(row['approved_index']) == row['approved_index_sha256'], 'Individual approval changed.')
        for entry in load_json(row['approved_index']).values():
            verified_reviewed_masks(entry, np.asarray(Image.open(entry['target_image']).convert('RGB')))
    result = dict(verified_utc=utc_now(), study='exploratory_pending_data_review', server_gate='not_eligible',
                  completed_stages=expected, all_training_logs_and_checkpoint_tensors_finite=True,
                  stage_log_sha256=stage_logs,
                  pinned_inputs_and_pipeline_unchanged=True, reports_recomputed_equal=True,
                  primary_report_sha256=sha256(published / 'primary/summary.json'),
                  ablation_report_sha256=sha256(published / 'summary.json'),
                  source_identity_sha256=sha256(root / 'status.json'),
                  initial_rgb_and_lpips_equal=True, all_views_retained=True,
                  verified_predictions_per_model=24, fixed_reviewed_masks=3,
                  additional_reviewed_masks=count, additional_masks_used_in_this_comparison=False,
                  gallery_sha256=sha256(gallery), reviewed_views=reviewed,
                  a0_reference=dict(metrics=summaries['A0']['group_weighted'],
                                    strata=strata(list(raw['A0'].values())),
                                    unchanged_original_predictions=24,
                                    evaluation_sha256=sha256(folders['A0'] / 'summary.json')),
                  comparisons_against_A0={n + '_vs_A0': comparison(
                      folders['A0'] / 'metrics.jsonl', folders[n] / 'metrics.jsonl', 17)
                      for n in ('A1', 'A2', 'A3')},
                  methods=recomputed['methods'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--work', required=True, help='Fresh ignored verification/gallery directory.')
    args = parser.parse_args()
    result = audit(args.output, args.work)
    print(json.dumps({k: result[k] for k in ('reports_recomputed_equal', 'comparisons_against_A0',
                                           'reviewed_views', 'additional_reviewed_masks')}, indent=2))
