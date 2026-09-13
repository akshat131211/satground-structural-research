"""Calibrate a predefined structural failure event; no correctness from agreement."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ResearchError, object_hash


def fit_calibration(rows, threshold=.03, score='disagreement'):
    if len(rows) < 50 or any(r.get('split') != 'calibration' for r in rows):
        raise ResearchError('Calibration requires >= 50 examples from the separate calibration split.')
    x = np.array([r[score] for r in rows], dtype=float)
    y = np.array([r['boundary_error'] > threshold for r in rows], dtype=int)
    if not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ResearchError('Calibration needs finite scores and both success and failure events.')
    fit = IsotonicRegression(out_of_bounds='clip').fit(x, y)
    return dict(score=score, failure_threshold=threshold, x=fit.X_thresholds_.tolist(), y=fit.y_thresholds_.tolist(),
                constant_probability=float(y.mean()), calibration_ids=sorted(r['sample_id'] for r in rows),
                calibration_groups=sorted({r['geo_group'] for r in rows}), data_hash=object_hash(rows))


def reliability_metrics(rows, calibration):
    if any(r['sample_id'] in calibration['calibration_ids'] or r['geo_group'] in calibration['calibration_groups'] for r in rows):
        raise ResearchError('Reliability evaluation overlaps calibration locations.')
    score = np.array([r[calibration['score']] for r in rows], dtype=float)
    failure = np.array([r['boundary_error'] > calibration['failure_threshold'] for r in rows], dtype=float)
    probability = np.interp(score, calibration['x'], calibration['y'])
    order = np.argsort(score, kind='stable')
    risk = np.cumsum(failure[order]) / np.arange(1, len(rows) + 1)
    ece = 0.0
    for low, high in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
        selected = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - failure[selected].mean())
    return dict(brier=float(np.mean((probability - failure) ** 2)), ece=float(ece),
                constant_brier=float(np.mean((calibration['constant_probability'] - failure) ** 2)),
                auroc=float(roc_auc_score(failure, score)) if len(np.unique(failure)) == 2 else None,
                average_precision=float(average_precision_score(failure, score)) if failure.any() else None,
                risk_coverage=[dict(coverage=float((i + 1) / len(rows)), risk=float(risk[i])) for i in range(len(rows))],
                count=len(rows), failure_threshold=calibration['failure_threshold'])
