"""Separate label-review effects from changes to the saved image generator."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from satground.common import ResearchError, load_json, read_jsonl, save_json, sha256, utc_now
from satground.metrics import grouped_mean


def report(index_path, output):
    index = load_json(index_path)
    if len(index) != 1:
        raise ResearchError('This onboarding diagnostic expects exactly one approved view.')
    sample_id, annotation = next(iter(index.items()))
    valid = np.asarray(Image.open(annotation['valid_mask'])) == 255
    if sha256(annotation['valid_mask']) != annotation['valid_mask_sha256']:
        raise ResearchError('Reviewed validity mask changed.')
    result = dict(updated_utc=utc_now(), purpose='Label-review diagnostic on fixed saved predictions',
        reviewed_views=1, manually_drawn_from_scratch=False, annotation_method=annotation['annotation_method'],
        review_blinded_to_model_predictions=annotation['review_blinded_to_model_predictions'],
        validity_fraction_of_reviewed_view=float(valid.mean()),
        valid_pixels_in_reviewed_view=int(valid.sum()), manual_index_sha256=sha256(index_path),
        model_retrained=False, improvement_claim=False, server_decision='insufficient_evidence',
        full_data_qa_complete=False, runs={})
    for name in ('GJ', 'GD'):
        old_folder = Path(f'runs/geometry100-{name}-validation-eval')
        new_folder = Path(f'runs/geometry100-{name}-validation-human1-eval')
        old, new = load_json(old_folder / 'summary.json'), load_json(new_folder / 'summary.json')
        if old['generation_sha256'] != new['generation_sha256'] or old['manifest_sha256'] != new['manifest_sha256']:
            raise ResearchError('Label probe must keep identical predictions and evaluation views.')
        if new.get('manual_index_sha256') != sha256(index_path) or new['manually_reviewed_count'] != 1:
            raise ResearchError('Reviewed index does not match the new evaluation.')
        if new['count'] != 24 or not new['mixed_manual_and_pseudo_labels']:
            raise ResearchError('Keep all 24 preselected views and disclose mixed labels.')
        for key in ('segmenter', 'segmenter_revision'):
            if old[key] != new[key]:
                raise ResearchError('Evaluation segmenter changed.')
        a = {r['sample_id']: r for r in read_jsonl(old_folder / 'metrics.jsonl')}
        b = {r['sample_id']: r for r in read_jsonl(new_folder / 'metrics.jsonl')}
        if set(a) != set(b) or sample_id not in b:
            raise ResearchError('Review comparison view sets differ.')
        if b[sample_id]['label_provenance'] != 'manually_reviewed':
            raise ResearchError('The selected view was not evaluated with reviewed masks.')
        for sid in a:
            keys = ('lpips', 'ssim', 'psnr') if sid == sample_id else ('lpips', 'ssim', 'psnr', 'boundary_error', 'building_iou')
            for key in keys:
                if a[sid][key] is None and b[sid][key] is None:
                    continue
                if not math.isclose(a[sid][key], b[sid][key], rel_tol=1e-6, abs_tol=1e-6):
                    raise ResearchError('A result outside the intended target-label revision changed.')
        result['runs'][name] = dict(saved_generation_sha256=new['generation_sha256'],
            evaluated_views=new['count'], reviewed_views=1, pseudo_label_views=23,
            reviewed_view_metrics={k: b[sample_id][k] for k in ('boundary_error', 'building_iou', 'lpips')},
            previous_pseudo_metrics_for_same_view={k: a[sample_id][k] for k in ('boundary_error', 'building_iou')},
            descriptive_mixed_label_aggregate=new['group_weighted'],
            evaluation_source_hash=new['evaluation_provenance']['pipeline_source_hash'])
    result['interpretation'] = ('Differences from earlier scores come from the reviewed target mask and validity region, '
        'not a changed model. One previously inspected view with partial visibility is insufficient for an accuracy claim. '
        'The same reviewed masks are applied to both models; all other views retain their original pseudo-label scoring.')
    out = Path(output)
    save_json(out / 'summary.json', result)
    lines = ['# One-view reviewed-label diagnostic', '', result['interpretation'], '',
        f"Reviewed scoring covers **{result['validity_fraction_of_reviewed_view']:.1%}** of this image. "
        'The remaining pixels are excluded because of occlusion, ambiguity, or dynamic objects. '
        'The annotation was assisted by a segmenter and explicit assistant corrections, then reviewed by the user. '
        'This was not a blind annotation study.', '',
        '| Saved model | Reviewed-view boundary error | Reviewed-view building IoU | Same-view LPIPS |',
        '|---|---:|---:|---:|']
    for name, run in result['runs'].items():
        m = run['reviewed_view_metrics']
        lines.append(f"| {name} | {m['boundary_error']:.6f} | {m['building_iou']:.6f} | {m['lpips']:.6f} |")
    lines += ['', 'The 24-view reruns contain one reviewed target and 23 pseudo-labelled targets. '
              'They remain descriptive mixed-label outputs, not a human-verified benchmark. '
              'No main data-QA flag or server gate was passed.', '',
              'See [mask review instructions](../../docs/MASK_REVIEW_GUIDE.md) and [aggregate evidence](summary.json).', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Verified label-review diagnostic:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', default='reports/reviewed-label-probe')
    args = parser.parse_args()
    report(args.index, args.output)
