"""Training-only contour support from nearby confident static-class anchors."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .common import ResearchError
from .losses import soft_boundary

MODE = 'anchored_contour_v1'
DEFINITION = dict(mode=MODE, radius_pixels=4, confidence_threshold=.7,
                  excluded_bridge_classes=[8, *range(11, 19)],
                  unsafe_halo_pixels=1, normalization='mean_over_union_support',
                  target='unchanged_hard_building_morphological_gradient_3x3')


def contour_support(building, valid, classes, confidence):
    arrays = [np.asarray(a) for a in (building, valid, classes, confidence)]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ResearchError('Contour inputs must be aligned two-dimensional arrays.')
    b, v, c, q = arrays
    if (not all(np.isin(a, [0, 1]).all() for a in (b, v)) or
            not np.isin(c, np.arange(19)).all() or not np.isfinite(q).all() or
            np.any((q < 0) | (q > 1)) or not np.array_equal(b, c == 2)):
        raise ResearchError('Invalid contour label values.')
    expected_valid = (q >= DEFINITION['confidence_threshold']) & (c < 11)
    if not np.array_equal(v, expected_valid):
        raise ResearchError('Contour v1 requires unchanged 0.7-confidence training labels.')

    def tensor(a):
        return torch.as_tensor(a.astype(np.float32))[None, None]

    def dilate(a, radius):
        return F.max_pool2d(a, radius * 2 + 1, 1, radius)

    bt, vt = tensor(b), tensor(v)
    edge = soft_boundary(bt) > 0
    legacy = -F.max_pool2d(-vt, 3, 1, 1) > 0
    # Vegetation and moving-object pixels cannot anchor or bridge a new contour.
    safe = (c < 11) & (c != 8)
    confident = q >= DEFINITION['confidence_threshold']
    building_anchor = tensor(confident & (c == 2))
    other_anchor = tensor(confident & safe & (c != 2))
    near_both = (dilate(building_anchor, 4) > 0) & (dilate(other_anchor, 4) > 0)
    safe_neighborhood = -F.max_pool2d(-tensor(safe), 3, 1, 1) > 0
    bridge = edge & near_both & safe_neighborhood
    support = legacy | bridge
    return dict(boundary_valid=support[0, 0].numpy(),
                legacy_valid=legacy[0, 0].numpy(),
                added_positive=(bridge & ~legacy)[0, 0].numpy(),
                target_boundary=edge[0, 0].numpy())


def validate_contour_labels(record, directory, rows):
    from pathlib import Path
    if record.get('boundary_definition') != DEFINITION or record.get('human_verified') is not False:
        raise ResearchError('Missing or changed contour pseudo-label definition.')
    if set(record['label_sha256']) != {r['sample_id'] for r in rows}:
        raise ResearchError('Contour label cohort differs from training.')
    if any(r['split'] != 'train' for r in rows):
        raise ResearchError('Contour labels must use training views only.')
    for row in rows:
        with np.load(Path(directory) / (row['sample_id'] + '.npz')) as data:
            expected = contour_support(*(data[k] for k in ('building', 'valid', 'classes', 'confidence')))
            if not np.array_equal(data['boundary_valid'], expected['boundary_valid']):
                raise ResearchError('Cached contour support differs from frozen definition.')
