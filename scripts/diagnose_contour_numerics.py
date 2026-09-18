"""Post-run numerical checks on a fixed training view; never take optimizer steps."""
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from satground.backend import load_style
from satground.camera import crop_panorama
from satground.common import (ResearchError, load_json, read_jsonl, require_development,
    safe_data_path, save_json, seed_everything, sha256, utc_now)
from satground.losses import Objective
from satground.runs import backend_from, read_config, validate_labels
from satground.semantics import rgb_tensor


def difference(reference, value):
    a, b = reference.detach().double().cpu(), value.detach().double().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ResearchError('Non-finite diagnostic gradient.')
    return dict(exact=torch.equal(a, b), relative_l2=float((b-a).norm()/a.norm().clamp_min(1e-12)),
                max_absolute=float((b-a).abs().max()), norm=float(b.norm()))


def diagnose(output):
    out = Path(output)
    if out.exists():
        raise ResearchError('Choose a fresh diagnostic directory.')
    cfg = read_config('configs/contour500/A2.yaml')
    rows = read_jsonl(cfg['train_manifest'])
    require_development(rows, {'train'})
    record = validate_labels(cfg['labels'], cfg['train_manifest'], 'data/vigor')
    out.mkdir(parents=True)
    seed_everything(cfg['seed'])
    backend = backend_from(cfg)
    backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
    objective = Objective(cfg['region_weight'], 0., cfg['perceptual_weight'], record['revision'], 'legacy')
    params = tuple(backend.adapter.parameters())
    before = {k: v.detach().clone() for k,v in backend.adapter.state_dict().items()}
    seed_everything(cfg['seed'])
    row = rows[random.randrange(len(rows))]
    raw = backend.cached_features(safe_data_path('data/vigor', row['satellite']), cfg['feature_cache'])
    with Image.open(safe_data_path('data/vigor', row['panorama'])) as im:
        target = rgb_tensor(crop_panorama(im.convert('RGB'), row['yaw'], row['pitch'], row['fov'], cfg['size']), 'cuda')
    with np.load(Path(cfg['labels']) / (row['sample_id']+'.npz')) as data:
        labels = {k: torch.from_numpy(data[k].astype(np.float32)).cuda()[None,None]
                  for k in ('building','valid','boundary_valid')}
    prediction = backend.render(raw, row, adapted=True)
    probe = prediction.detach().requires_grad_(True)
    image_gradients, losses = {}, {}
    for mode in ('legacy','anchored_contour_v1'):
        objective.boundary_mode = mode
        loss, values = objective(probe, target, labels)
        image_gradients[mode] = torch.autograd.grad(loss, probe)[0].detach()
        losses[mode] = values
    fixed_gradient = image_gradients['legacy']
    vectors = []
    for _ in range(5):
        grads = torch.autograd.grad(prediction, params, grad_outputs=fixed_gradient, retain_graph=True)
        vectors.append(torch.cat([g.detach().reshape(-1).cpu() for g in grads]))
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            torch.use_deterministic_algorithms(True, warn_only=True)
            grads = torch.autograd.grad(prediction, params, grad_outputs=fixed_gradient)
            warning_messages = sorted({str(w.message) for w in observed})
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    backend.assert_frozen()
    unchanged = all(torch.equal(v,before[k]) for k,v in backend.adapter.state_dict().items())
    if not unchanged or any(p.grad is not None for p in params):
        raise ResearchError('Probe changed parameters or stored gradients.')
    result = dict(created_utc=utc_now(), optimizer_steps=0, training_views=1, backward_repeats=5,
        source_hash=load_json('runs/contour500-seed17/status.json')['identity']['pipeline_source_hash'],
        script_sha256=sha256(__file__), torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32),
        losses=losses, image_gradient_comparison=difference(image_gradients['legacy'],image_gradients['anchored_contour_v1']),
        same_graph_fixed_upstream_repeats=[difference(vectors[0],v) for v in vectors[1:]],
        deterministic_warnings=warning_messages, parameters_unchanged=unchanged,
        stored_gradients_absent=True, frozen_models_verified=True,
        limitations=['One fixed training view at the initial adapter; not a training replicate.',
            'Observed nondeterminism does not prove every historical difference has one cause.'])
    save_json(out/'summary.json', result)
    # Local-only provenance: no source identifiers enter the aggregate summary.
    save_json(out/'local-identity.json',dict(sample_id=row['sample_id'],manifest_sha256=sha256(cfg['train_manifest'])))
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    diagnose(parser.parse_args().output)
