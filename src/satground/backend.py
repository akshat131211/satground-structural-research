"""Unmodified pinned Sat3DGen plus a shared, differentiable feature adapter."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .adapter import StructuralAdapter
from .camera import camera_matrices, satellite_tensor
from .common import ROOT, MODEL_REV, ResearchError, check_vendor, load_json, object_hash, sha256


class Sat3DBackend:
    def __init__(self, size=128, chunk_rows=4, checkpoint_rays=True, amp=False, width=16, field_mode=None):
        if not torch.cuda.is_available():
            raise ResearchError('The pinned Sat3DGen renderer requires an NVIDIA CUDA device.')
        if size < 32 or size % 2 or chunk_rows < 1:
            raise ResearchError('Resolution must be even and >= 32; chunk_rows must be positive.')
        vendor = check_vendor()
        sys.path.insert(0, str(vendor))
        from source.generator import Sat3DGen
        self.size, self.chunk_rows, self.checkpoint_rays, self.amp = size, chunk_rows, checkpoint_rays, amp
        if field_mode not in (None, 'density', 'joint'):
            raise ResearchError('Training field mode must be density or joint, or absent for the original renderer.')
        self.field_mode = field_mode
        model_path = ROOT / 'artifacts' / 'Sat3DGen'
        if not (model_path / 'diffusion_pytorch_model.safetensors').exists():
            raise ResearchError('Missing pretrained checkpoint. Run bootstrap.')
        lock = load_json(ROOT / 'artifacts' / 'asset-lock.json')
        locked = {Path(r['file'].replace('\\', '/')).name: r['sha256'] for r in lock['files']}
        for name in ('config.json', 'diffusion_pytorch_model.safetensors'):
            if name not in locked or sha256(model_path / name) != locked[name]:
                raise ResearchError(f'Pinned model asset failed its checksum: {name}')
        Sat3DGen._skip_backbone_weights = True
        self.model = Sat3DGen.from_pretrained(str(model_path), local_files_only=True,
                                             low_cpu_mem_usage=False).eval().requires_grad_(False)
        if self.model.opt.representation_type != 'oneplane' or self.model.opt.network.triplane.dim != 32:
            raise ResearchError('This adapter integration requires the pinned oneplane configuration.')
        self.model.point_sampler_definition(size)
        # Keep encoder on CPU until a tile needs encoding; frozen renderer stays on GPU.
        for name, module in self.model.named_children():
            if name != 'unet_model':
                module.cuda()
        self.adapter = StructuralAdapter(64, width).cuda()
        self.w_sky = self.sky = None

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.float16, enabled=self.amp)

    def set_style(self, style):
        value = torch.as_tensor(style, dtype=torch.float32, device='cuda').reshape(1, 270)
        if not torch.isfinite(value).all():
            raise ResearchError('Illumination code contains non-finite values.')
        with torch.no_grad(), self.autocast():
            self.w_sky = self.model.w_sky_prepare(value).detach()
            self.sky = self.model.w_sky2sky_feature_2D(self.w_sky, value).detach()

    @torch.no_grad()
    def encode(self, image):
        self.model.unet_model.cuda()
        try:
            with self.autocast():
                result = self.model.unet_model(image.cuda()).float().detach()
        finally:
            self.model.unet_model.cpu()
            torch.cuda.empty_cache()
        return result

    def cached_features(self, image_path, cache_dir):
        key = object_hash(dict(image_sha256=sha256(image_path), model=MODEL_REV, amp=self.amp,
                               preprocessing='RGB-bicubic256-ImageNet', storage='float32'))
        path = Path(cache_dir) / f'{key}.pt'
        if path.is_file():
            features = torch.load(path, map_location='cuda', weights_only=True)
        else:
            features = self.encode(satellite_tensor(image_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            torch.save(features.cpu(), temp)
            os.replace(temp, path)
        if tuple(features.shape) != (1, 64, 320, 320) or not torch.isfinite(features).all():
            raise ResearchError('Invalid cached scene features.')
        return features

    def render(self, features, camera, adapted=False):
        if self.field_mode is not None:
            from .fields import render_fields
            mode = self.field_mode if adapted else 'base'
            return render_fields(self, features, camera, (mode,))[mode]['rgb']
        from source.xyz2thetaphi import xyz2thetaphi
        if self.sky is None:
            raise ResearchError('A fixed training-only illumination code is required.')
        c2w, _, intrinsics = camera_matrices(camera['h_offset'], camera['w_offset'], camera['yaw'],
                                            camera['pitch'], camera['fov'], self.size)
        c2w = self.model.c2w_prepare(torch.from_numpy(c2w).unsqueeze(0).cuda().clone())
        # Ray inverses and coordinates stay float32 even under autocast.
        with torch.autocast('cuda', enabled=False):
            samples = self.model.point_sampler_per(torch.from_numpy(intrinsics).unsqueeze(0).cuda(), c2w)
        with self.autocast():
            raw = self.adapter(features) if adapted else features
            planes = (raw[:, :32].unsqueeze(1), raw[:, 32:].mean(dim=2))
            chunks = []
            for start in range(0, self.size // 2, self.chunk_rows):
                selected = {key: samples[key][:, start:start + self.chunk_rows]
                            for key in ('points_world', 'rays_world', 'radii', 'ray_origins')}
                def integrate(plane, line, selected=selected):
                    result = self.model.from_point_sampling2result(selected, [plane, line], self.w_sky)
                    return result.feature_raw, result.alpha_raw
                if adapted and torch.is_grad_enabled() and self.checkpoint_rays:
                    chunk = checkpoint(integrate, *planes, use_reentrant=False, preserve_rng_state=True)
                else:
                    chunk = integrate(*planes)
                chunks.append(chunk)
            feature_image = torch.cat([x[0] for x in chunks], dim=2)
            alpha = torch.cat([x[1] for x in chunks], dim=2)
            sky = F.grid_sample(self.sky, xyz2thetaphi(samples.rays_world), align_corners=True).clamp(0, 1)
            # SR includes instance normalization: always process the full stitched image.
            return self.model.sr_module(feature_image * alpha + sky * (1 - alpha)).float()

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ResearchError('A frozen base-model parameter was modified or received gradients.')


def load_style(path, train_manifest_hash=None):
    record = load_json(path)
    if record.get('source_split') != 'train' or record.get('synthetic', False):
        raise ResearchError('Scientific runs require a real, training-only illumination code.')
    if train_manifest_hash and record.get('manifest_sha256') != train_manifest_hash:
        raise ResearchError('Illumination code belongs to a different training manifest.')
    import numpy as np
    value = np.asarray(record['histogram'], dtype=float)
    if value.shape != (270,) or not np.isfinite(value).all() or (value < 0).any():
        raise ResearchError('Invalid training illumination histogram.')
    if not np.allclose(value.reshape(3, 90).sum(axis=1), 1, atol=1e-5):
        raise ResearchError('Every training illumination channel must sum to one.')
    return record['histogram']
