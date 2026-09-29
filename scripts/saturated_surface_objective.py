"""Saturated valid-surface revision, separately preserving the failed v1 method.

The concept was fixed in docs/VALID_SURFACE_SATURATION_REVISION.md before this
implementation: w=(1+min(d,8))/9 across valid components with observed interfaces.
An unobserved component remains unsupported. Inputs, validity rules, distances
within radius8, class normalization, probability clamp and coefficient are kept.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_propagation
import torch

from valid_surface_objective import (
    COEFFICIENT as V1_COEFFICIENT, MASS_FLOOR as V1_MASS_FLOOR,
    RADIUS as V1_RADIUS, _array_hash, _binary,
    prepare_surface_labels as prepare_v1_surface_labels,
    valid_surface_loss as v1_surface_runtime,
)


POLICY = 'valid_surface_saturated_v2'
RADIUS = 8
MASS_FLOOR = 32
COEFFICIENT = .5
VALIDITY_POLICY = (
    'same v1 3x3 validity erosion with outside-image invalid; opposite '
    '4-neighbour endpoints; 8-neighbour geodesics without diagonal corner '
    'cutting; min(distance,8) saturates throughout observable seeded components; '
    'zero only outside domain or in components without observed interfaces'
)
_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def prepare_surface_labels(building, valid):
    """Prepare CPU labels without changing source masks or readonly v1 arrays.

    Four-connectivity gives exactly the reachability of eight-neighbour paths
    without corner cutting: every permitted diagonal has valid orthogonal steps.
    V1 supplies true shortest distances through8; farther reachable distances
    need only the fixed cap8. This avoids computing unused large metric distances.
    ``distance`` is therefore min(d,8), not a claim of uncapped metric distance;
    infinity denotes unsupported locations only. No whole-component support is
    called a band: near-band and saturated/total support counts remain distinct.
    """
    if (V1_RADIUS, V1_MASS_FLOOR, V1_COEFFICIENT) != (RADIUS, MASS_FLOOR, COEFFICIENT):
        raise ValueError('Original policy constants differ; do not silently change the revision.')
    target, accepted = _binary(building, 'Building mask'), _binary(valid, 'Validity mask')
    if target.shape != accepted.shape:
        raise ValueError('Building and validity masks must have the same shape.')
    old = prepare_v1_surface_labels(target, accepted)
    domain, interface = old['domain'].copy(), old['interface'].copy()
    near_distances = old['distance']
    reachable = binary_propagation(interface, structure=_CONNECTIVITY, mask=domain)
    near_band = domain & np.isfinite(near_distances)
    if (near_band & ~reachable).any():
        raise ValueError('A v1 distance source became unreachable; validity rules differ.')
    distance = np.full(target.shape, np.inf, dtype=np.float64)
    distance[reachable] = RADIUS
    distance[near_band] = near_distances[near_band]
    weight = np.zeros(target.shape, dtype=np.float32)
    weight[reachable] = ((1 + distance[reachable]) / (1 + RADIUS)).astype(np.float32)
    weights = np.stack((weight * ~target, weight * target)).astype(np.float32)
    masses = weights.sum(axis=(1, 2), dtype=np.float64)
    denominators = np.maximum(masses, MASS_FLOOR).astype(np.float32)
    # Source/interface counts retain their literal v1 meanings. Old band/mass
    # counts are deliberately not relabelled as current full-component support.
    source_keys = ('accepted_valid_pixels', 'domain_pixels', 'historical_positive_contour_pixels',
                   'interface_source_pixels', 'historical_contour_pixels_retained_as_sources',
                   'historical_contour_pixels_not_sources', 'domain_components',
                   'domain_components_without_observed_interface', 'surface_support_present')
    counts = {key: old['counts'][key] for key in source_keys}
    counts.update(near_band_pixels=int(near_band.sum()),
                  near_band_class_pixels=[int((near_band & (target == c)).sum()) for c in (0, 1)],
                  surface_scored_pixels=int(reachable.sum()),
                  saturated_distance_pixels=int((reachable & (distance >= RADIUS)).sum()),
                  beyond_radius_saturated_pixels=int((reachable & ~near_band).sum()),
                  class_scored_pixels=[int((reachable & (target == c)).sum()) for c in (0, 1)],
                  class_weight_mass=masses.tolist(), class_denominators=denominators.tolist(),
                  unsupported_components=old['counts']['domain_components_without_observed_interface'])
    provenance = dict(policy=POLICY, validity_policy=VALIDITY_POLICY, radius=RADIUS,
                      mass_floor=MASS_FLOOR, coefficient=COEFFICIENT,
                      building_array_sha256=_array_hash(target), valid_array_sha256=_array_hash(accepted),
                      weights_sha256=_array_hash(weights), experimental=True,
                      distance_kind='capped_shortest_path_distance; infinity only outside observed seeded components',
                      original_interface_policy=old['provenance']['policy'])
    for array in (weights, denominators, domain, interface, distance, reachable):
        array.setflags(write=False)
    return dict(weights=weights, denominators=denominators, domain=domain,
                interface=interface, distance=distance, reachable=reachable,
                counts=counts, provenance=provenance)


def surface_support_counts(building, valid):
    """Aggregate counts; no masks, positions, scenes or raw reviews returned."""
    prepared = prepare_surface_labels(building, valid)
    return dict(prepared['counts'], policy=POLICY, radius=RADIUS, mass_floor=MASS_FLOOR)


def valid_surface_loss(probability, prepared):
    """Reuse the unchanged class mean, floor, float32 runtime and clamp.

    The exact runtime expression is the v1 formula with separately prepared v2
    weights. No detached predictions or new coefficient is applied. Inputs must
    carry the v2 preparation provenance; old band weights are never relabelled.
    """
    provenance = prepared.get('provenance', {})
    if (provenance.get('policy') != POLICY or provenance.get('radius') != RADIUS or
            provenance.get('mass_floor') != MASS_FLOOR or provenance.get('coefficient') != COEFFICIENT):
        raise ValueError('Prepared labels do not carry the fixed saturated-v2 policy.')
    if isinstance(probability, torch.Tensor):
        if not bool(torch.isfinite(probability).all()) or not bool(((probability >= 0) & (probability <= 1)).all()):
            raise ValueError('Probability must be finite and lie in [0,1].')
    return v1_surface_runtime(probability, prepared)
