"""CPU-only, post-hoc boundary diagnostics; never change the primary fitting criterion."""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, maximum_filter, minimum_filter

from prepare_reviewed_fitting import require, verify_bundle
from satground.annotations import binary_mask
from satground.common import load_json, read_jsonl, save_json, sha256
from satground.metrics import building_metrics, grouped_mean


def boundary_counts(predicted, target, valid):
    """Mirror primary inner edges, then separately remove only the image perimeter."""
    predicted, target, valid = [np.asarray(value, bool) for value in (predicted, target, valid)]
    require(predicted.shape == target.shape == valid.shape and target.ndim == 2, 'Wrong diagnostic mask shapes.')
    require(valid.any(), 'No scored pixels.')
    predicted, target = predicted & valid, target & valid
    interior = binary_erosion(valid, structure=np.ones((3, 3)), border_value=1)
    edges = [(mask ^ binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)) & interior
             for mask in (predicted, target)]
    perimeter = np.zeros_like(valid)
    perimeter[[0, -1], :] = True
    perimeter[:, [0, -1]] = True
    pe, te = [edge & ~perimeter for edge in edges]
    if not pe.any() and not te.any():
        distance = 0.0
    elif not pe.any() or not te.any():
        distance = 1.0
    else:
        distance = float(.5 * (distance_transform_edt(~pe)[te].mean() + distance_transform_edt(~te)[pe].mean())
                         / np.hypot(*target.shape))
    return dict(primary_target_edge_pixels=int(edges[1].sum()),
                primary_predicted_edge_pixels=int(edges[0].sum()),
                target_perimeter_edge_pixels=int((edges[1] & perimeter).sum()),
                predicted_perimeter_edge_pixels=int((edges[0] & perimeter).sum()),
                target_nonperimeter_edge_pixels=int(te.sum()), predicted_nonperimeter_edge_pixels=int(pe.sum()),
                nonperimeter_boundary_error=distance,
                nonperimeter_empty_case='both_empty' if not pe.any() and not te.any() else
                                         'one_empty' if not pe.any() or not te.any() else 'neither_empty')


def diagnose(bundle, run, output, models):
    bundle, run, output = Path(bundle), Path(run), Path(output)
    require(not output.exists(), 'Use a fresh diagnostic output; preserve earlier audits.')
    record = verify_bundle(bundle)
    rows = read_jsonl(bundle / 'manifest.jsonl')
    manual = load_json(bundle / 'manual-index.json')
    require(models and len(models) == len(set(models)) and set(models) <= {'A0', 'A2', 'A3'}, 'Wrong model list.')
    private_rows, support, sources = {}, [], {}
    for i, row in enumerate(rows):
        entry = manual[row['sample_id']]
        shape = (row['size'], row['size'])
        building, valid = binary_mask(entry['building_mask'], shape), binary_mask(entry['valid_mask'], shape)
        training_edges = maximum_filter(building, size=3, mode='nearest') != minimum_filter(building, size=3, mode='nearest')
        training_valid = minimum_filter(valid, size=3, mode='nearest')
        support.append(dict(fitting_view=i + 1, positive_training_boundary_pixels=int((training_edges & training_valid).sum()),
                            training_boundary_valid_pixels=int(training_valid.sum())))
    for name in models:
        folder = run / f'{name}-eval'
        summary = load_json(folder / 'summary.json')
        scored = read_jsonl(folder / 'metrics.jsonl')
        require(summary['manifest_sha256'] == record['fitting_manifest_sha256']
                and summary['manual_index_sha256'] == sha256(bundle / 'manual-index.json'), 'Diagnostic scoring identity differs.')
        scores = {row['sample_id']: row for row in scored}
        require(len(scores) == len(scored) == len(rows) and set(scores) == set(manual), 'Incomplete diagnostic cohort.')
        private_rows[name] = []
        for i, row in enumerate(rows):
            sid = row['sample_id']
            require(scores[sid]['split'] == 'train', 'Diagnostics are restricted to the fitting training views.')
            entry = manual[sid]
            shape = (row['size'], row['size'])
            target, valid = binary_mask(entry['building_mask'], shape), binary_mask(entry['valid_mask'], shape)
            prediction = binary_mask(folder / 'masks' / f'{sid}.png', shape)
            primary = building_metrics(prediction, target, valid)
            require(all(primary[key] == scores[sid][key] for key in primary), 'Primary metric no longer reproduces.')
            private_rows[name].append(dict(scores[sid], fitting_view=i + 1, **boundary_counts(prediction, target, valid)))
        sources[name] = dict(metrics_sha256=sha256(folder / 'metrics.jsonl'), summary_sha256=sha256(folder / 'summary.json'))
    first = private_rows[models[0]]
    edge_total = sum(r['primary_target_edge_pixels'] for r in first)
    perimeter_total = sum(r['target_perimeter_edge_pixels'] for r in first)
    positive = sum(r['positive_training_boundary_pixels'] for r in support)
    valid_total = sum(r['training_boundary_valid_pixels'] for r in support)
    result = dict(scope='posthoc_training_fitting_diagnostic_only', primary_metric_and_criterion_unchanged=True,
        model_names=models, fitting_views=len(rows), sources_sha256=sources, bundle_sha256=sha256(bundle / 'bundle.json'),
        positive_training_boundary_pixels=positive, training_boundary_valid_pixels=valid_total,
        positive_training_boundary_fraction=positive / valid_total,
        primary_target_edge_pixels=edge_total, target_perimeter_edge_pixels=perimeter_total,
        target_perimeter_edge_fraction=perimeter_total / edge_total if edge_total else None,
        models={name: dict(primary_boundary_error=grouped_mean(values, 'boundary_error'),
                          nonperimeter_boundary_error=grouped_mean(values, 'nonperimeter_boundary_error'),
                          predicted_perimeter_edge_pixels=sum(r['predicted_perimeter_edge_pixels'] for r in values),
                          nonperimeter_empty_counts={case: sum(r['nonperimeter_empty_case'] == case for r in values)
                                                    for case in ('both_empty', 'one_empty', 'neither_empty')})
                for name, values in private_rows.items()},
        per_view=[dict(support[i], primary_target_edge_pixels=first[i]['primary_target_edge_pixels'],
                       target_perimeter_edge_pixels=first[i]['target_perimeter_edge_pixels'],
                       models={name: {key: values[i][key] for key in ('boundary_error', 'nonperimeter_boundary_error',
                                'building_iou', 'lpips', 'nonperimeter_empty_case')} for name, values in private_rows.items()})
                  for i in range(len(rows))],
        held_out_generalization_evidence=False, server_gate_eligible=False,
        diagnostic_timing='Defined after frozen A0 evaluation while the predeclared fitting pair was active; not a selection metric.')
    save_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='data/review/reviewed-fitting8-v1')
    parser.add_argument('--run', default='runs/reviewed-fitting200-seed17')
    parser.add_argument('--models', nargs='+', required=True, choices=['A0', 'A2', 'A3'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    diagnose(args.bundle, args.run, args.output, args.models)
