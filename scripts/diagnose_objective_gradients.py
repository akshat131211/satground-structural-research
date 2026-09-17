"""Measure existing objective gradients on training images without taking updates."""
import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from satground.backend import load_style
from satground.common import (ResearchError, load_json, object_hash, provenance, read_jsonl,
    require_development, safe_data_path, save_json, seed_everything, sha256, utc_now)
from satground.camera import crop_panorama
from satground.losses import Objective, soft_boundary, structural_losses
from satground.manifests import assert_disjoint
from satground.runs import backend_from, read_config, validate_labels
from satground.semantics import rgb_tensor


TERMS = ('rgb', 'lpips', 'region', 'boundary')


def vector_summary(vectors):
    vectors = {k: v.detach().double().reshape(-1).cpu() for k, v in vectors.items()}
    if any(not torch.isfinite(v).all() for v in vectors.values()):
        raise ResearchError('Non-finite component gradient.')
    vectors['A1'] = vectors['rgb'] + vectors['lpips']
    vectors['A2'] = vectors['A1'] + vectors['region']
    vectors['A3'] = vectors['A2'] + vectors['boundary']
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    cosines = {}
    for a, b in (('boundary', 'region'), ('boundary', 'rgb'), ('boundary', 'lpips'),
                 ('boundary', 'A1'), ('boundary', 'A2'), ('A2', 'A3')):
        denominator = norms[a] * norms[b]
        cosines[a + '_vs_' + b] = float(torch.dot(vectors[a], vectors[b]) / denominator) if denominator else None
    ratios = {k: norms['boundary'] / norms[k] if norms[k] else None for k in ('region', 'A1', 'A2')}
    return dict(weighted_norms=norms, cosines=cosines, boundary_norm_ratios=ratios)


def component_losses(objective, prediction, target, labels):
    probability = objective.segmenter.probabilities(prediction)[:, 2:3]
    region, boundary = structural_losses(probability, labels['building'], labels['valid'])
    return dict(rgb=F.l1_loss(prediction, target),
                lpips=objective.perceptual(prediction * 2 - 1, target * 2 - 1).mean(),
                region=region, boundary=boundary)


def verify_gradient_sum(actual, expected, tolerance=5e-5):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ResearchError('Non-finite objective gradient comparison.')
    difference = float((actual - expected).double().norm())
    norm = float(actual.double().norm())
    relative = difference / max(norm, 1e-12)
    if relative > tolerance:
        raise ResearchError(f'Diagnostic gradient differs from actual Objective: {relative:g}')
    return relative


def diagnose(output, limit=None):
    out = Path(output)
    if out.exists():
        raise ResearchError('Choose a fresh diagnosis directory.')
    cfg_path = Path('configs/conservative500/A3.yaml')
    cfg = read_config(cfg_path)
    rows = read_jsonl(cfg['train_manifest'])
    validation = read_jsonl(cfg['validation_manifest'])
    require_development(rows, {'train'}); require_development(validation, {'validation'})
    assert_disjoint(rows + validation)
    if len(rows) != 100:
        raise ResearchError('Expected the original 100-view training cohort.')
    rows = sorted(rows, key=lambda r: r['sample_id'])
    if limit is not None:
        if limit < 1 or limit > 100:
            raise ResearchError('Invalid explicit smoke-test limit.')
        rows = rows[:limit]
    labels_record = validate_labels(cfg['labels'], cfg['train_manifest'], 'data/vigor')
    checkpoints = {step: Path(f'runs/conservative500-seed17/A3-train/step-{step:06d}.pt') for step in (300, 500)}
    files = [cfg_path, Path(cfg['train_manifest']), Path(cfg['validation_manifest']), Path(cfg['style']),
             Path(cfg['labels']) / 'provenance.json', Path(__file__), Path('docs/STRUCTURAL_DIAGNOSIS.md'),
             *checkpoints.values()]
    files += [Path(cfg['labels']) / (r['sample_id'] + '.npz') for r in rows]
    files += [safe_data_path('data/vigor', r[k]) for r in rows for k in ('satellite', 'panorama')]
    identity = dict(files={str(p): sha256(p) for p in files}, pipeline_source_hash=provenance()['pipeline_source_hash'])
    out.mkdir(parents=True)
    weights = dict(rgb=1., lpips=cfg['perceptual_weight'], region=cfg['region_weight'], boundary=cfg['boundary_weight'])
    state = dict(created_utc=utc_now(), identity=identity, status='initializing', completed=0,
                 expected=len(rows) * 3, optimizer_steps=0, training_views=len(rows), weights=weights,
                 checkpoints=[0, 300, 500], smoke_test=limit is not None)
    save_json(out / 'status.json', state)
    begin = time.perf_counter()
    try:
        seed_everything(cfg['seed'])
        # Decomposition adds separately backpropagated terms. Disable TF32 to
        # bound reduction noise before comparing their sum with one backward pass.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        state['tf32_disabled_for_gradient_decomposition'] = True
        backend = backend_from(cfg)
        initial = {k: v.detach().clone() for k, v in backend.adapter.state_dict().items()}
        backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
        objective = Objective(cfg['region_weight'], cfg['boundary_weight'], cfg['perceptual_weight'], labels_record['revision'])
        params = tuple(backend.adapter.parameters())
        torch.cuda.reset_peak_memory_stats()
        for step in (0, 300, 500):
            if step:
                checkpoint = torch.load(checkpoints[step], weights_only=True, map_location='cpu')
                if checkpoint['config'] != cfg or checkpoint['step'] != step:
                    raise ResearchError('Checkpoint config/step mismatch.')
                backend.adapter.load_state_dict(checkpoint['adapter'])
            else:
                backend.adapter.load_state_dict(initial)
            before = {k: v.detach().clone() for k, v in backend.adapter.state_dict().items()}
            for row in rows:
                state.update(status='running', current_step=step)
                raw = backend.cached_features(safe_data_path('data/vigor', row['satellite']), cfg['feature_cache'])
                with Image.open(safe_data_path('data/vigor', row['panorama'])) as im:
                    target = rgb_tensor(crop_panorama(im.convert('RGB'), row['yaw'], row['pitch'], row['fov'], cfg['size']), 'cuda')
                with np.load(Path(cfg['labels']) / (row['sample_id'] + '.npz')) as label:
                    labels = {k: torch.from_numpy(label[k].astype(np.float32)).cuda()[None, None] for k in ('building', 'valid')}
                seed_everything(int(object_hash([cfg['render_seed'], row['sample_id']])[:8], 16))
                prediction = backend.render(raw, row, adapted=True)
                probe = prediction.detach().requires_grad_(True)
                losses = component_losses(objective, probe, target, labels)
                image_gradients = {}
                for name in TERMS:
                    image_gradients[name] = torch.autograd.grad(weights[name] * losses[name], probe,
                                                               retain_graph=name != TERMS[-1])[0].detach()
                actual_loss, values = objective(probe, target, labels)
                for name in TERMS:
                    if not math.isclose(float(losses[name].detach()), values[name], rel_tol=1e-6, abs_tol=1e-7):
                        raise ResearchError('Component values differ from existing Objective.')
                expected_total = sum(weights[k] * float(losses[k].detach()) for k in TERMS)
                if not math.isclose(float(actual_loss.detach()), expected_total, rel_tol=1e-6, abs_tol=1e-7):
                    raise ResearchError('Weighted components differ from existing Objective.')
                actual_gradient = torch.autograd.grad(actual_loss, probe)[0].detach()
                sum_error = verify_gradient_sum(actual_gradient, sum(image_gradients.values()))
                adapter_gradients = {}
                for name in TERMS:
                    gradients = torch.autograd.grad(prediction, params, grad_outputs=image_gradients[name],
                                                    retain_graph=True, allow_unused=False)
                    adapter_gradients[name] = torch.cat([g.detach().reshape(-1) for g in gradients]).cpu()
                actual_adapter = torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(
                    prediction, params, grad_outputs=actual_gradient)]).cpu()
                adapter_sum_error = verify_gradient_sum(actual_adapter, sum(adapter_gradients.values()), tolerance=2e-4)
                interior = -F.max_pool2d(-labels['valid'], 3, stride=1, padding=1)
                target_boundary = soft_boundary(labels['building']) * interior
                record = dict(sample_id=row['sample_id'], geo_group=row['geo_group'], city=row['city'], split='train',
                    checkpoint_step=step, raw_losses=values, image_gradients=vector_summary(image_gradients),
                    adapter_gradients=vector_summary(adapter_gradients), image_gradient_sum_relative_error=sum_error,
                    adapter_gradient_sum_relative_error=adapter_sum_error,
                    valid_pixels=int(labels['valid'].sum()), interior_valid_pixels=int(interior.sum()),
                    scored_building_pixels=int((labels['building'] * labels['valid']).sum()),
                    target_boundary_pixels=int((target_boundary > 0).sum()))
                with (out / 'records.jsonl').open('a', encoding='utf-8') as stream:
                    import json
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                backend.assert_frozen()
                if any(p.grad is not None for p in params) or any(p.grad is not None or p.requires_grad
                    for module in (objective.segmenter.model, objective.perceptual) for p in module.parameters()):
                    raise ResearchError('Unexpected stored or trainable reference gradients.')
                state.update(completed=state['completed'] + 1, elapsed_seconds=time.perf_counter() - begin,
                             peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20, updated_utc=utc_now())
                save_json(out / 'status.json', state)
                if state['completed'] % 10 == 0 or state['completed'] == 1:
                    print(f"Probe {state['completed']}/{state['expected']}, A3 step {step}, finite gradients, "
                          f"{state['elapsed_seconds']:.1f}s, {state['peak_vram_mib']:.1f} MiB", flush=True)
                del prediction, probe, actual_loss, losses, raw, target, labels, gradients, image_gradients, adapter_gradients
            if any(not torch.equal(v, before[k]) for k, v in backend.adapter.state_dict().items()):
                raise ResearchError('Gradient probe changed adapter parameters.')
        if any(sha256(p) != h for p, h in identity['files'].items()) or provenance()['pipeline_source_hash'] != identity['pipeline_source_hash']:
            raise ResearchError('Source or input changed during diagnosis.')
        state.update(status='complete', current_step=None, parameters_unchanged=True, frozen_models_verified=True,
                     records_sha256=sha256(out / 'records.jsonl'), updated_utc=utc_now())
        save_json(out / 'status.json', state)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=utc_now())
        save_json(out / 'status.json', state)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--smoke-limit', type=int)
    a = p.parse_args()
    diagnose(a.output, a.smoke_limit)
