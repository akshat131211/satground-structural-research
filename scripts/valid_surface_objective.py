"""Exploratory CPU-prepared surface supervision, separate from historical losses.

The accepted masks are inputs, never repaired. Only opposite-label interfaces
observable inside an eroded validity domain provide distance sources. Neither
crop borders nor unknown occlusions close a building. The fixed eight-pixel band
does not supervise distant errors or boundary-free components: unchanged region
BCE must still control those regions. This implementation is not method selection
or a claim of novelty. It adapts established distance/surface supervision:
https://proceedings.mlr.press/v102/kervadec19a.html
https://arxiv.org/html/1904.10030
"""
from __future__ import annotations

import hashlib
import heapq
import math

import numpy as np
from scipy.ndimage import binary_erosion, label, maximum_filter, minimum_filter
import torch


POLICY = 'valid_surface_excess_v1'
RADIUS = 8
MASS_FLOOR = 32
COEFFICIENT = 0.5
VALIDITY_POLICY = (
    '3x3 validity erosion with outside-image invalid; opposite 4-neighbour '
    'endpoints within that domain; 8-neighbour geodesics with no diagonal corner '
    'cutting; radius 8; class-weight mass floor 32; no invented unknown/crop edges'
)


def _binary(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not array.size or not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must be a nonempty two-dimensional binary array.')
    return array.astype(bool, copy=True)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    header = f'{array.dtype.str}:{array.shape}:'.encode('ascii')
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _interfaces(target, domain):
    seeds = np.zeros_like(domain)
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]),
                 (np.s_[:, :-1], np.s_[:, 1:])):
        interface = domain[a] & domain[b] & (target[a] != target[b])
        seeds[a] |= interface
        seeds[b] |= interface
    return seeds


def _distances(domain, seeds):
    """Multi-source shortest paths; infinite means unreachable or beyond radius."""
    height, width = domain.shape
    distances = np.full(domain.shape, np.inf, dtype=np.float64)
    queue = [(0.0, int(r), int(c)) for r, c in np.argwhere(seeds)]
    distances[seeds] = 0
    heapq.heapify(queue)
    neighbours = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                  (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                  (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
    while queue:
        distance, r, c = heapq.heappop(queue)
        if distance != distances[r, c]:
            continue
        for dr, dc, cost in neighbours:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width and domain[nr, nc]):
                continue
            if dr and dc and not (domain[r, nc] and domain[nr, c]):
                continue
            candidate = distance + cost
            if candidate <= RADIUS and candidate < distances[nr, nc]:
                distances[nr, nc] = candidate
                heapq.heappush(queue, (candidate, nr, nc))
    return distances


def prepare_surface_labels(building, valid):
    """Return private spatial weights plus aggregate support/provenance metadata.

    Source arrays are copied, not mutated. Distances are intentionally infinite
    outside the observed band; the runtime weights and denominators are finite.
    Each class keeps its fixed normalizer even if one class has no usable support.
    """
    target, accepted = _binary(building, 'Building mask'), _binary(valid, 'Validity mask')
    if target.shape != accepted.shape:
        raise ValueError('Building and validity masks must have the same shape.')
    domain = binary_erosion(accepted, structure=np.ones((3, 3)), border_value=0)
    seeds = _interfaces(target, domain)
    distances = _distances(domain, seeds)
    band = domain & np.isfinite(distances)
    weight = np.zeros(target.shape, dtype=np.float32)
    weight[band] = ((1 + distances[band]) / (1 + RADIUS)).astype(np.float32)
    weights = np.stack((weight * ~target, weight * target)).astype(np.float32)
    masses = weights.sum(axis=(1, 2), dtype=np.float64)
    denominators = np.maximum(masses, MASS_FLOOR).astype(np.float32)

    # Historical max/min pooling treats out-of-image neighbours as absent.
    legacy_domain = minimum_filter(accepted, size=3, mode='nearest')
    legacy_edges = (maximum_filter(target, size=3, mode='nearest') !=
                    minimum_filter(target, size=3, mode='nearest')) & legacy_domain
    components, component_count = label(domain)  # 4-connectivity matches valid paths.
    seeded_components = np.unique(components[seeds])
    counts = dict(
        accepted_valid_pixels=int(accepted.sum()), domain_pixels=int(domain.sum()),
        historical_positive_contour_pixels=int(legacy_edges.sum()),
        interface_source_pixels=int(seeds.sum()),
        historical_contour_pixels_retained_as_sources=int((legacy_edges & seeds).sum()),
        historical_contour_pixels_not_sources=int((legacy_edges & ~seeds).sum()),
        band_pixels=int(band.sum()), class_band_pixels=[int((band & (target == c)).sum()) for c in (0, 1)],
        class_weight_mass=masses.tolist(), class_denominators=denominators.tolist(),
        domain_components=int(component_count),
        domain_components_without_observed_interface=int(component_count - len(seeded_components)),
        surface_support_present=bool(seeds.any()),
    )
    provenance = dict(policy=POLICY, validity_policy=VALIDITY_POLICY, radius=RADIUS,
                      mass_floor=MASS_FLOOR, coefficient=COEFFICIENT,
                      building_array_sha256=_array_hash(target), valid_array_sha256=_array_hash(accepted),
                      weights_sha256=_array_hash(weights), experimental=True)
    for array in (weights, denominators, domain, seeds, distances):
        array.setflags(write=False)
    return dict(weights=weights, denominators=denominators, domain=domain,
                interface=seeds, distance=distances, counts=counts, provenance=provenance)


def surface_support_counts(building, valid):
    """Aggregate helper: no image/coordinate/scene identity is returned."""
    prepared = prepare_surface_labels(building, valid)
    return dict(prepared['counts'], policy=POLICY, radius=RADIUS, mass_floor=MASS_FLOOR)


def valid_surface_loss(probability, prepared):
    """Differentiable float32 surface excess for one view; no prediction detach.

    Accept HxW or singleton-prefix shapes such as 1x1xHxW. The unclipped formula
    is half the sum of class-normalized weighted |p-y|; use historical probability
    clipping [1e-6,1-1e-6]. Exact binary predictions therefore retain its negligible
    clipping residual. Return the unweighted term; COEFFICIENT is the declared
    future objective coefficient, not silently applied here.
    """
    if not isinstance(probability, torch.Tensor) or not probability.is_floating_point():
        raise ValueError('Probability must be a floating-point Torch tensor.')
    weights = np.asarray(prepared['weights'])
    denominators = np.asarray(prepared['denominators'])
    if (weights.ndim != 3 or weights.shape[0] != 2 or denominators.shape != (2,) or
            tuple(probability.shape[-2:]) != tuple(weights.shape[1:]) or
            probability.numel() != weights.shape[1] * weights.shape[2]):
        raise ValueError('Probability and prepared labels must describe one matching view.')
    if (not np.isfinite(weights).all() or not np.isfinite(denominators).all() or
            (weights < 0).any() or (weights > 1).any() or (denominators < MASS_FLOOR).any()):
        raise ValueError('Prepared surface weights or denominators are invalid.')
    if not bool(torch.isfinite(probability).all()):
        raise ValueError('Probability contains nonfinite values.')
    p = probability.float().reshape(weights.shape[1:]).clamp(1e-6, 1 - 1e-6)
    w = torch.tensor(weights, device=p.device, dtype=torch.float32)
    denominator = torch.tensor(denominators, device=p.device, dtype=torch.float32)
    return 0.5 * ((w[0] * p).sum() / denominator[0] +
                  (w[1] * (1 - p)).sum() / denominator[1])
