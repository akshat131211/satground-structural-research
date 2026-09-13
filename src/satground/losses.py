from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_boundary(probability):
    # Morphological gradient remains differentiable almost everywhere.
    maximum = F.max_pool2d(probability, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-probability, 3, stride=1, padding=1)
    return maximum - minimum


def structural_losses(building_probability, building_target, valid):
    probability = building_probability.clamp(1e-6, 1 - 1e-6)
    denominator = valid.sum().clamp_min(1)
    region = (F.binary_cross_entropy(probability, building_target, reduction='none') * valid).sum() / denominator
    # Do not supervise boundaries adjacent to ignored pixels.
    interior_valid = -F.max_pool2d(-valid, 3, stride=1, padding=1)
    boundary = ((soft_boundary(probability) - soft_boundary(building_target)).abs() * interior_valid).sum()
    boundary = boundary / interior_valid.sum().clamp_min(1)
    return region, boundary


class Objective:
    def __init__(self, region_weight=0, boundary_weight=0, perceptual_weight=.1, segmenter_revision=None):
        import lpips
        self.perceptual = lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
        self.region_weight, self.boundary_weight, self.perceptual_weight = region_weight, boundary_weight, perceptual_weight
        self.segmenter = None
        if region_weight or boundary_weight:
            from .semantics import Segmenter
            self.segmenter = Segmenter(revision=segmenter_revision)

    def __call__(self, prediction, target, labels=None):
        rgb = F.l1_loss(prediction, target)
        perceptual = self.perceptual(prediction * 2 - 1, target * 2 - 1).mean()
        total = rgb + self.perceptual_weight * perceptual
        parts = dict(rgb=rgb, lpips=perceptual)
        if self.segmenter:
            if labels is None:
                raise ValueError('Structural objectives require cached pseudo-labels.')
            probability = self.segmenter.probabilities(prediction)[:, 2:3]
            region, boundary = structural_losses(probability, labels['building'], labels['valid'])
            total = total + self.region_weight * region + self.boundary_weight * boundary
            parts.update(region=region, boundary=boundary)
        parts['total'] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}
