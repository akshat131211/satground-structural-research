"""Structural metrics with explicit empty-mask handling and geographic inference."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .common import ResearchError


def building_metrics(predicted, target, valid=None):
    predicted, target = np.asarray(predicted, bool), np.asarray(target, bool)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ResearchError('Building masks must have matching H x W shape.')
    valid = np.ones_like(target) if valid is None else np.asarray(valid, bool)
    if not valid.any():
        raise ResearchError('Building metrics have no valid pixels.')
    predicted, target = predicted & valid, target & valid
    union = (predicted | target).sum()
    iou = float((predicted & target).sum() / union) if union else 1.0
    interior = binary_erosion(valid, structure=np.ones((3, 3)), border_value=1)
    pe = (predicted ^ binary_erosion(predicted, structure=np.ones((3, 3)), border_value=0)) & interior
    te = (target ^ binary_erosion(target, structure=np.ones((3, 3)), border_value=0)) & interior
    if not pe.any() and not te.any():
        distance = 0.0
    elif not pe.any() or not te.any():
        distance = 1.0  # A missing/extra building is penalized, never silently excluded.
    else:
        distance = .5 * (distance_transform_edt(~pe)[te].mean() + distance_transform_edt(~te)[pe].mean())
        distance /= math.hypot(*target.shape)
    return dict(building_iou=iou, boundary_error=float(distance), target_building_pixels=int(target.sum()),
                predicted_building_pixels=int(predicted.sum()), valid_pixels=int(valid.sum()))


def paired_group_bootstrap(control, proposed, key='boundary_error', repeats=2000, seed=17):
    a, b = {r['sample_id']: r for r in control}, {r['sample_id']: r for r in proposed}
    if set(a) != set(b) or not a:
        raise ResearchError('Paired comparisons require identical nonempty sample IDs.')
    groups = defaultdict(list)
    for sample in a:
        if a[sample]['geo_group'] != b[sample]['geo_group']:
            raise ResearchError('Geographic group mismatch.')
        groups[a[sample]['geo_group']].append(float(a[sample][key]) - float(b[sample][key]))
    if len(groups) < 2:
        raise ResearchError('At least two independent geographic groups are needed for a confidence interval.')
    # Equal-weight group means; nearby/cropped views never count as independent resamples.
    means = np.array([np.mean(values) for values in groups.values()])
    rng = np.random.default_rng(seed)
    estimates = np.mean(rng.choice(means, (repeats, len(means)), replace=True), axis=1)
    return dict(mean_improvement=float(means.mean()), ci95=np.quantile(estimates, [.025, .975]).tolist(),
                geographic_groups=len(groups), samples=len(a), weighting='equal_geographic_group')


def grouped_mean(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row['geo_group']].append(float(row[key]))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def server_gate(comparisons, review):
    """Missing human review or provenance can never produce a pass."""
    missing = []
    for field in ('human_review_complete', 'no_frequent_new_hallucinations', 'provenance_complete', 'audit_frozen_before_evaluation'):
        if review.get(field) is not True:
            missing.append(field)
    if len(comparisons) != 3 or len({c['seed'] for c in comparisons}) != 3:
        missing.append('three_distinct_training_seeds')
    if missing:
        return dict(status='insufficient_evidence', missing=missing)
    required = {'relative_boundary_reduction', 'boundary_ci95', 'iou_change', 'relative_lpips_change'}
    if any(not required.issubset(c) for c in comparisons):
        return dict(status='insufficient_evidence', missing=['complete_metrics'])
    checks = dict(boundary_gain=all(c['relative_boundary_reduction'] >= .10 for c in comparisons),
                  confidence_interval=all(c['boundary_ci95'][0] > 0 for c in comparisons),
                  iou=all(c['iou_change'] >= -.01 for c in comparisons),
                  lpips=all(c['relative_lpips_change'] <= .05 for c in comparisons))
    return dict(status='pass' if all(checks.values()) else 'fail', checks=checks,
                interpretation='Conservative gate: every seed must satisfy thresholds.')
