"""Density/appearance interventions using frozen-baseline importance samples.

This is a diagnostic and a geometry-only adaptation control, not an instance
field or a new source of geometric ground truth. All modes use the same 3D
coordinates; the density/appearance switch happens before volume integration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .camera import camera_matrices
from .common import ResearchError

FIELD_MODES = ('base', 'density', 'appearance', 'joint')


def render_fields(backend, features, camera, modes=FIELD_MODES):
    from source.rendering.utils import sample_importance, unify_attributes
    from source.xyz2thetaphi import xyz2thetaphi

    if not modes or len(set(modes)) != len(modes) or any(m not in FIELD_MODES for m in modes):
        raise ResearchError('Choose unique base/density/appearance/joint field interventions.')
    if backend.sky is None:
        raise ResearchError('A fixed training-only illumination code is required.')
    model = backend.model
    if model.point_integrator.density_noise_std:
        raise ResearchError('Shared field interventions require noise-free density integration.')
    c2w, _, intrinsics = camera_matrices(camera['h_offset'], camera['w_offset'], camera['yaw'],
                                        camera['pitch'], camera['fov'], backend.size)
    c2w = model.c2w_prepare(torch.from_numpy(c2w)[None].cuda().clone())
    with torch.autocast('cuda', enabled=False):
        samples = model.point_sampler_per(torch.from_numpy(intrinsics)[None].cuda(), c2w)

    def planes(raw):
        return (raw[:, :32].unsqueeze(1), raw[:, 32:].mean(dim=2))

    with backend.autocast():
        base_planes = planes(features.detach())
        needs_adapter = any(m != 'base' for m in modes)
        adapted_planes = planes(backend.adapter(features.detach())) if needs_adapter else base_planes
        chunks = []
        for start in range(0, backend.size // 2, backend.chunk_rows):
            selected = {key: samples[key][:, start:start + backend.chunk_rows]
                        for key in ('points_world', 'rays_world', 'radii', 'ray_origins')}

            def integrate(plane, line, selected=selected):
                n, h, w, k, _ = selected['points_world'].shape
                rays = h * w
                points = selected['points_world'].reshape(n, rays, k, 3)
                radii = selected['radii'].reshape(n, rays, k, 1)

                def query(coords, representation):
                    count = coords.shape[-2]
                    value = model.mlp(model.point_representer(
                        coords.reshape(n, -1, 3), ref_representation=list(representation)), backend.w_sky)
                    return {key: value[key].reshape(n, rays, count, -1) for key in ('density', 'color')}

                # The proposal is always frozen A0, including in geometry-only training.
                with torch.no_grad():
                    base_coarse = query(points, base_planes)
                    if model.num_importance:
                        coarse = model.point_integrator(base_coarse['color'], base_coarse['density'], radii)
                        fine_radii = sample_importance(radii, coarse['weight'], model.num_importance,
                                                       smooth_weights=True)
                        origins = selected['ray_origins']
                        origins = origins[:, None].expand(-1, rays, -1) if origins.ndim == 2 else origins.reshape(n, rays, 3)
                        fine_points = origins[:, :, None] + fine_radii * selected['rays_world'].reshape(n, rays, 1, 3)
                        base_fine = query(fine_points, base_planes)
                if needs_adapter:
                    adapted_coarse = query(points, (plane, line))
                    if model.num_importance:
                        adapted_fine = query(fine_points, (plane, line))

                outputs = []
                for mode in modes:
                    density = adapted_coarse['density'] if mode in ('density', 'joint') else base_coarse['density']
                    color = adapted_coarse['color'] if mode in ('appearance', 'joint') else base_coarse['color']
                    distances = radii
                    if model.num_importance:
                        fine_density = adapted_fine['density'] if mode in ('density', 'joint') else base_fine['density']
                        fine_color = adapted_fine['color'] if mode in ('appearance', 'joint') else base_fine['color']
                        distances, color, density = unify_attributes(radii, color, density,
                                                                     fine_radii, fine_color, fine_density)
                    result = model.point_integrator(color, density, distances)
                    for key in ('composite_color', 'opacity', 'composite_radial_dist'):
                        outputs.append(result[key].reshape(n, h, w, -1).permute(0, 3, 1, 2).contiguous())
                return tuple(outputs)

            if needs_adapter and torch.is_grad_enabled() and backend.checkpoint_rays:
                chunks.append(checkpoint(integrate, *adapted_planes, use_reentrant=False, preserve_rng_state=True))
            else:
                chunks.append(integrate(*adapted_planes))
        sky = F.grid_sample(backend.sky, xyz2thetaphi(samples.rays_world), align_corners=True).clamp(0, 1)
        outputs = {}
        for index, mode in enumerate(modes):
            feature, opacity, radial = [torch.cat([c[index * 3 + j] for c in chunks], dim=2) for j in range(3)]
            rgb = model.sr_module(feature * opacity + sky * (1 - opacity)).float()
            outputs[mode] = dict(rgb=rgb, opacity=opacity.float(), radial=radial.float())
        return outputs
