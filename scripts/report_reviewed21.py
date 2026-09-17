"""Audit fixed predictions under expanded, explicitly versioned human labels."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_exploratory500 import require, verified_predictions
from report_checkpoint_diagnosis import strata
from satground.annotations import verified_reviewed_masks
from satground.common import load_json, read_jsonl, sha256, utc_now
from satground.metrics import grouped_mean, paired_group_bootstrap


METRICS = ('boundary_error', 'building_iou', 'lpips', 'ssim')


def verify_label_only_change(old, new, reviewed_ids):
    a, b = ({r['sample_id']: r for r in records} for records in (old, new))
    require(len(a) == len(old) and len(b) == len(new) and set(a) == set(b), 'Evaluation cohort changed.')
    for sid, current in b.items():
        prior = a[sid]
        for key in ('geo_group', 'split', 'lpips', 'psnr', 'ssim', 'entropy', 'identical_rgb'):
            require(current[key] == prior[key], 'Non-label metric or image identity changed: ' + key)
        if sid not in reviewed_ids or prior['label_provenance'] == 'manually_reviewed':
            require(current == prior, 'An unchanged target record differs.')


def delta(control, proposed):
    base = grouped_mean(control, 'boundary_error')
    quality = grouped_mean(control, 'lpips')
    require(base > 0 and quality > 0, 'Relative change requires positive baseline errors.')
    return dict(relative_boundary_reduction=(base-grouped_mean(proposed, 'boundary_error'))/base,
                boundary_bootstrap=paired_group_bootstrap(control, proposed, seed=17),
                building_iou_change=grouped_mean(proposed, 'building_iou')-grouped_mean(control, 'building_iou'),
                relative_lpips_change=(grouped_mean(proposed, 'lpips')-quality)/quality)


def report(root, index_path, output, local_output):
    root, index_path, output, local_output = map(Path, (root, index_path, output, local_output))
    require(not output.exists() and not local_output.exists(), 'Use fresh report output paths.')
    state = load_json(root / 'status.json')
    require(state['status'] == 'complete' and state['completed'] == ['A0', 'A1', 'A2', 'A3'],
            'All four re-evaluations must finish.')
    require(all(sha256(p) == h for p, h in state['identity']['files'].items()), 'Re-scoring inputs changed.')
    index = load_json(index_path)
    require(len(index) == 21, 'Expected original three plus eighteen reviewed targets.')
    require(len({e['review_view_number'] for e in index.values()}) == len(index), 'Duplicate review number.')
    paths = {'A0': Path('runs/exploratory500-A0-reference'), **{
        n: Path('runs/conservative500-seed17') / f'{n}-validation' for n in ('A1', 'A2', 'A3')}}
    models, raw, ref = {}, {}, None
    for name, folder in paths.items():
        evaluation = root / (name + '-eval')
        new = load_json(evaluation / 'summary.json')
        rows = read_jsonl(evaluation / 'metrics.jsonl')
        old_dir = Path(str(folder) + '-eval')
        old = load_json(old_dir / 'summary.json')
        previous = read_jsonl(old_dir / 'metrics.jsonl')
        previous_by_id = {r['sample_id']: r for r in previous}
        require(len(rows) == 24 and all(r['split'] == 'validation' for r in rows), 'Wrong development cohort.')
        require(new['generation'] == old['generation'] == load_json(folder / 'generation.json'),
                'Generation metadata changed.')
        require(new['generation_sha256'] == old['generation_sha256'] == sha256(folder / 'generation.json'),
                'Generation identity changed.')
        require(new['segmenter'] == old['segmenter'] and new['segmenter_revision'] == old['segmenter_revision'],
                'Evaluation segmenter changed.')
        require(new['manual_index_sha256'] == sha256(index_path) and new['manually_reviewed_count'] == 21,
                'Reviewed labels mismatch.')
        require(new['evaluation_provenance']['pipeline_source_hash'] == state['identity']['pipeline_source_hash'],
                'Evaluation source differs from re-scoring identity.')
        require(set(verified_predictions(folder)) == {r['sample_id'] for r in rows}, 'Prediction cohort mismatch.')
        verify_label_only_change(previous, rows, set(index))
        for row in rows:
            sid = row['sample_id']
            for sub in ('targets', 'masks'):
                require(sha256(evaluation / sub / (sid+'.png')) == sha256(old_dir / sub / (sid+'.png')),
                        'Target pixels or predicted segmentation changed.')
            if sid in index:
                target = np.asarray(Image.open(evaluation / 'targets' / (sid+'.png')).convert('RGB'))
                b, v = verified_reviewed_masks(index[sid], target)
                require(row['target_building_pixels'] == int((b & v).sum()) and
                        row['valid_pixels'] == int(v.sum()) and row['label_provenance'] == 'manually_reviewed',
                        'Reviewed pixels mismatch.')
        labels = {r['sample_id']: tuple(r[k] for k in
                  ('geo_group', 'target_building_pixels', 'valid_pixels', 'label_provenance')) for r in rows}
        if ref is None:
            ref = labels
        require(ref == labels, 'Models use different scoring labels.')
        means = {m: grouped_mean(rows, m) for m in METRICS}
        require(all(math.isclose(means[m], new['group_weighted'][m], abs_tol=1e-12) for m in METRICS),
                'Published aggregate differs from per-image records.')
        models[name] = dict(metrics=means, strata=strata(rows), original_metrics=old['group_weighted'],
            evaluation_sha256=sha256(evaluation/'summary.json'), metrics_sha256=sha256(evaluation/'metrics.jsonl'),
            unchanged_prediction_images=24, unchanged_predicted_segmentations=24,
            unchanged_image_quality_records=24, old_labels_changed=sum(
                a['target_building_pixels'] != b['target_building_pixels'] or a['valid_pixels'] != b['valid_pixels']
                for b in rows for a in (previous_by_id[b['sample_id']],)))
        raw[name] = rows
    comparisons = {}
    for a, b in [('A3', 'A1'), ('A2', 'A1'), ('A3', 'A2'), ('A3', 'A0')]:
        comparisons[a+'_vs_'+b] = dict(all24=delta(raw[b], raw[a]), reviewed21=delta(
            [r for r in raw[b] if r['sample_id'] in index], [r for r in raw[a] if r['sample_id'] in index]))
    order = sorted(index.items(), key=lambda pair: pair[1]['review_view_number'])
    lookup = {n: {r['sample_id']: r for r in rows} for n, rows in raw.items()}
    reviewed = [dict(review_view_number=entry['review_view_number'], additional_batch_view=entry.get('review_batch_view_number'),
        metrics={n: {m: lookup[n][sid][m] for m in METRICS} for n in raw},
        scored_building_pixels=lookup['A0'][sid]['target_building_pixels'],
        valid_pixels=lookup['A0'][sid]['valid_pixels']) for sid, entry in order]
    wins = {}
    for proposed, control in [('A3', 'A1'), ('A3', 'A2'), ('A3', 'A0')]:
        changes = [r['metrics'][control]['boundary_error']-r['metrics'][proposed]['boundary_error'] for r in reviewed]
        wins[proposed+'_vs_'+control] = dict(improved=sum(x>0 for x in changes),
            worsened=sum(x<0 for x in changes), tied=sum(x==0 for x in changes))
    result = dict(created_utc=utc_now(),study='post_hoc_expanded_human_label_diagnostic',server_gate='not_eligible',
        new_training_performed=False,original_results_preserved=True,validation_views=24,reviewed_views=21,
        pseudo_label_views=3,manual_index_sha256=sha256(index_path),models=models,comparisons=comparisons,
        reviewed_boundary_directions=wins,
        source_identity_sha256=sha256(root/'status.json'),annotation_quality_followup_batch_views=[9,16],
        limitations=['Previously inspected development images, one training seed, incomplete pairing and footprint QA.',
                     'User-reviewed machine proposals; no independent annotator or instance-level certification.',
                     'Possible transport-structure taxonomy ambiguity in batch view 9; submitted masks retained.',
                     'Changed labels alter evaluation, not model predictions; three targets still have pseudo-labels.'])
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    lines=['# Expanded-label evaluation of unchanged predictions','',
           '21 reviewed targets and three remaining pseudo-labelled targets; all 24 development views retained.',
           'These measurements use a new label version. The original experiment remains unchanged.','',
           '| Model | All-view boundary | Reviewed boundary | All-view IoU | Reviewed IoU | LPIPS |',
           '|---|---:|---:|---:|---:|---:|']
    for name, model in models.items():
        m=model['metrics']; r=model['strata']['reviewed']['metrics']
        lines.append(f"| {name} | {m['boundary_error']:.6f} | {r['boundary_error']:.6f} | {m['building_iou']:.6f} | {r['building_iou']:.6f} | {m['lpips']:.6f} |")
    lines += ['', 'Each geographic group has equal weight within the reported cohort. LPIPS is unchanged by target-mask review.',
              '', '[Detailed paired intervals and strata](summary.json). [Interpretation](interpretation.md).', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    local_output.mkdir(parents=True)
    (local_output/'reviewed-view-metrics.json').write_text(
        json.dumps(reviewed,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    for start in range(0,len(order),4):
        selected=order[start:start+4]; canvas=Image.new('RGB',(5*192,len(selected)*220),'#17202a'); draw=ImageDraw.Draw(canvas)
        for i,(sid,entry) in enumerate(selected):
            items=[('Target '+str(entry['review_view_number']),Path(entry['target_image']))]+[(n,paths[n]/(sid+'.png')) for n in raw]
            for col,(label,path) in enumerate(items):
                with Image.open(path) as image: canvas.paste(image.convert('RGB').resize((192,192)),(col*192,i*220+23))
                draw.text((col*192+4,i*220+4),label,fill='white')
        canvas.save(local_output/f'page-{start//4+1}.png')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/conservative500-reviewed21')
    p.add_argument('--index',default='data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json')
    p.add_argument('--output',required=True)
    p.add_argument('--local-output',required=True)
    a=p.parse_args(); result=report(a.root,a.index,a.output,a.local_output)
    print(json.dumps(dict(comparisons=result['comparisons'],reviewed_boundary_directions=result['reviewed_boundary_directions']),indent=2))
