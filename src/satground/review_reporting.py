"""Describe label corrections without attributing them to a model change."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import verified_reviewed_masks
from .common import ResearchError, load_json, read_jsonl, save_json, sha256, utc_now


def verify_label_only_change(old, new, old_rows, new_rows, reviewed_ids, index_hash):
    """Reject changed predictions, view sets, or metrics unrelated to reviewed labels."""
    reviewed_ids = set(reviewed_ids)
    a = {r['sample_id']: r for r in old_rows}
    b = {r['sample_id']: r for r in new_rows}
    if (not reviewed_ids or len(a) != len(old_rows) or len(b) != len(new_rows)
            or set(a) != set(b) or not reviewed_ids.issubset(b)
            or old['count'] != len(a) or new['count'] != len(b)):
        raise ResearchError('Review comparison must retain every unique original view.')
    for key in ('generation_sha256', 'manifest_sha256', 'segmenter', 'segmenter_revision'):
        if not old.get(key) or old[key] != new.get(key):
            raise ResearchError('Predictions, manifest, or evaluation segmenter changed: ' + key)
    if old.get('manually_reviewed_count', 0):
        raise ResearchError('Use the original pseudo-label evaluation as the reference.')
    if (new.get('manual_index_sha256') != index_hash
            or new.get('manually_reviewed_count') != len(reviewed_ids)
            or new.get('mixed_manual_and_pseudo_labels') != (len(reviewed_ids) < len(b))):
        raise ResearchError('Reviewed-index provenance or label counts do not match.')
    actual_reviewed = {sid for sid, row in b.items() if row['label_provenance'] == 'manually_reviewed'}
    if actual_reviewed != reviewed_ids:
        raise ResearchError('Per-view reviewed labels do not match the approved index.')
    for sid in a:
        for key in ('geo_group', 'split'):
            if a[sid][key] != b[sid][key]:
                raise ResearchError('A view changed geographic group or split.')
        keys = ['lpips', 'ssim', 'psnr']
        if sid not in reviewed_ids:
            keys += ['boundary_error', 'building_iou']
            if a[sid]['label_provenance'] != b[sid]['label_provenance']:
                raise ResearchError('An unreviewed view changed label provenance.')
        for key in keys:
            x, y = a[sid][key], b[sid][key]
            if x is None and y is None:
                continue
            if x is None or y is None or not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6):
                raise ResearchError('A metric outside the intended label change changed: ' + key)
    return b


def report_reviewed_cohort(index_path, evaluation_tag, output, runs_root='runs'):
    if not evaluation_tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in evaluation_tag):
        raise ResearchError('Evaluation tag must be a simple lowercase folder label.')
    out = Path(output)
    if out.exists():
        raise ResearchError('Preserve earlier reports; choose a fresh output directory.')
    index = load_json(index_path)
    if not index:
        raise ResearchError('At least one approved mask pair is required.')
    public_views, ordered = [], []
    numbers = [e.get('review_view_number') for e in index.values()]
    if any(type(n) is not int or n < 1 for n in numbers) or len(set(numbers)) != len(numbers):
        raise ResearchError('Each reviewed view needs a unique positive review number.')
    for sid, entry in sorted(index.items(), key=lambda pair: pair[1]['review_view_number']):
        with Image.open(entry['target_image']) as im:
            target = np.asarray(im.convert('RGB'))
        _, valid = verified_reviewed_masks(entry, target)
        ordered.append(sid)
        public_views.append(dict(review_view_number=entry['review_view_number'],
            valid_pixels=int(valid.sum()), validity_fraction=float(valid.mean()),
            annotation_method=entry['annotation_method'], annotation_source=entry.get('proposal_provenance'),
            review_blinded_to_model_predictions=entry.get('review_blinded_to_model_predictions')))
    result = dict(updated_utc=utc_now(), purpose='Reviewed-label diagnostic on unchanged saved predictions',
        manual_index_sha256=sha256(index_path), reviewed_views=len(index), views=public_views, runs={},
        model_retrained=False, improvement_claim=False, full_data_qa_complete=False,
        server_decision='insufficient_evidence')
    manifests, source_hashes = set(), set()
    for name in ('GJ', 'GD'):
        old_path = Path(runs_root) / f'geometry100-{name}-validation-eval'
        new_path = Path(runs_root) / f'geometry100-{name}-validation-{evaluation_tag}-eval'
        old, new = load_json(old_path / 'summary.json'), load_json(new_path / 'summary.json')
        a, b = read_jsonl(old_path / 'metrics.jsonl'), read_jsonl(new_path / 'metrics.jsonl')
        checked = verify_label_only_change(old, new, a, b, index, result['manual_index_sha256'])
        manifests.add(new['manifest_sha256'])
        source_hashes.add(new['evaluation_provenance']['pipeline_source_hash'])
        metrics = []
        for sid in ordered:
            row = checked[sid]
            if row['valid_pixels'] != public_views[ordered.index(sid)]['valid_pixels']:
                raise ResearchError('Evaluation scored a different validity mask.')
            metrics.append(dict(review_view_number=index[sid]['review_view_number'],
                **{k: row[k] for k in ('boundary_error', 'building_iou', 'lpips', 'ssim', 'psnr')}))
        result['runs'][name] = dict(evaluated_views=new['count'], reviewed_views=len(index),
            pseudo_label_views=new['count']-len(index), saved_generation_sha256=new['generation_sha256'],
            evaluation_summary_sha256=sha256(new_path / 'summary.json'),
            evaluation_source_hash=new['evaluation_provenance']['pipeline_source_hash'],
            reviewed_view_metrics=metrics, descriptive_mixed_label_aggregate=new['group_weighted'])
    if len(manifests) != 1 or len(source_hashes) != 1:
        raise ResearchError('The two models need matching evaluation manifests and source versions.')
    result['interpretation'] = ('These are annotation-review diagnostics on fixed model outputs. Changes from '
        'earlier scores reflect changed target labels and validity regions. The small, previously inspected '
        'review set does not establish model improvement or complete the main data QA.')
    lines = [f'# Reviewed-label diagnostic: {len(index)} views', '', result['interpretation'], '',
        'GJ adapts density and appearance jointly; GD adapts density only. Lower boundary error and higher '
        'building IoU are preferable. Each view uses the same accepted masks for both models.', '',
        '| Review view | Scored pixels | GJ boundary | GD boundary | GJ IoU | GD IoU |',
        '|---|---:|---:|---:|---:|---:|']
    for i, view in enumerate(public_views):
        gj, gd = (result['runs'][n]['reviewed_view_metrics'][i] for n in ('GJ', 'GD'))
        lines.append(f"| {view['review_view_number']} | {view['validity_fraction']:.1%} | "
            f"{gj['boundary_error']:.6f} | {gd['boundary_error']:.6f} | "
            f"{gj['building_iou']:.6f} | {gd['building_iou']:.6f} |")
    run = result['runs']['GJ']
    lines += ['', f"Each rerun retains all {run['evaluated_views']} preselected views: "
        f"{len(index)} reviewed targets and {run['pseudo_label_views']} pseudo-labelled targets. "
        'The mixed aggregate is descriptive and is not a human-verified benchmark.', '',
        'Checks confirm unchanged predictions, view sets, segmenter revisions, all image-quality metrics, '
        'and structural metrics for every unreviewed view. Model-generated building masks still come from '
        'the evaluation segmenter. Human review corrects target labels, not predicted masks.', '',
        'Masks are reviewed edits of machine proposals, with provenance retained for each view. '
        'Different validity coverage limits comparisons across views. No server gate has passed.', '',
        'See [aggregate evidence](summary.json), [the initial one-view report](../reviewed-label-probe/report.md), '
        'and [mask review instructions](../../docs/MASK_REVIEW_GUIDE.md).', '']
    save_json(out / 'summary.json', result)
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return result
