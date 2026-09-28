"""Bounded reviewed-mask adapter fitting, separate from historical pseudo-label training."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from prepare_reviewed_fitting import PROTOCOL, PROTOCOL_DOC, check_pins, make_configs, require, verify_bundle
from satground.backend import load_style
from satground.camera import crop_panorama
from satground.common import (ROOT, load_json, object_hash, provenance, read_jsonl, safe_data_path,
                             save_json, seed_everything, sha256, write_jsonl)
from satground.readiness import require_training_readiness
from satground.runs import atomic_checkpoint, backend_from, read_config, restore_rng, rng_state
from satground.semantics import rgb_tensor


def source_pins():
    files = [PROTOCOL, PROTOCOL_DOC, ROOT / 'program.md',
             *[ROOT / 'scripts' / name for name in ('prepare_reviewed_fitting.py', 'train_reviewed_fitting.py',
                 'run_reviewed_fitting.py', 'report_reviewed_fitting.py')]]
    return {str(p): sha256(p) for p in files}


def fitting_identity(config_path, data_root):
    cfg = read_config(config_path)
    require(cfg['experiment'] in ('A2', 'A3'), 'Only A2 and A3 may receive fitting updates.')
    bundle = Path(cfg['fitting_bundle'])
    record = verify_bundle(bundle)
    protocol = load_json(PROTOCOL)
    require(record['protocol'] == protocol, 'Bundle differs from the frozen fitting protocol.')
    expected = make_configs(bundle, protocol, cfg['train_manifest'], cfg['style'])[cfg['experiment']]
    require(cfg == expected, 'Configuration differs from the fixed 200-step fitting protocol.')
    require(str(Path(config_path).resolve()) in record['artifact_sha256'], 'Configuration is not frozen in the bundle.')
    rows = read_jsonl(cfg['fitting_manifest'])
    readiness = require_training_readiness(cfg, rows, cfg['steps'])
    verified = {r['path']: r['sha256'] for r in load_json('reports/data-readiness.json')['files']}
    hashes = {}
    for row in rows:
        for key in ('satellite', 'panorama'):
            path = row[key]
            if path not in hashes:
                hashes[path] = sha256(safe_data_path(data_root, path))
            require(verified.get(path) == hashes[path], 'Training RGB changed since data verification.')
            require(record['source_file_sha256'].get(str(safe_data_path(data_root, path))) == hashes[path],
                    'Training RGB differs from the fitting freeze.')
    p = provenance()
    identity = dict(config=cfg, bundle_sha256=sha256(bundle / 'bundle.json'),
                    bundle_inputs=record['source_file_sha256'], bundle_artifacts=record['artifact_sha256'],
                    source_file_sha256=source_pins(), pipeline_source_hash=p['pipeline_source_hash'],
                    environment={k: p[k] for k in ('python', 'packages', 'code_revision', 'model_revision')},
                    readiness=readiness, data_digest=object_hash(hashes),
                    parent_training_manifest_sha256=sha256(cfg['train_manifest']),
                    fitting_manifest_sha256=sha256(cfg['fitting_manifest']), style_sha256=sha256(cfg['style']))
    return cfg, rows, identity


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def prepare_resume(output, identity_hash, budget):
    """Validate first, then preserve only completed log tails superseded by recovery."""
    output = Path(output)
    state = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    require(state['identity_hash'] == identity_hash, 'Cannot resume altered fitting inputs, source or environment.')
    run = load_json(output / 'run.json')
    require(run['identity_hash'] == identity_hash == object_hash(run['identity'])
            and state['config'] == run['identity']['config'], 'Checkpoint and original run identity differ.')
    require(type(state['step']) is int and 0 < state['step'] <= budget, 'Invalid checkpoint update count.')
    require(all(key in state for key in ('adapter', 'optimizer', 'scaler', 'rng')) and finite_tree(state),
            'Incomplete or nonfinite fitting checkpoint.')
    log = read_jsonl(output / 'training.jsonl')
    require([r['step'] for r in log] == list(range(1, len(log) + 1)) and state['step'] <= len(log) <= budget
            and finite_tree(log), 'Training log does not support the recovery checkpoint.')
    tail = [r for r in log if r['step'] > state['step']]
    if tail:
        write_jsonl(output / f'interrupted-steps-{uuid4().hex}.jsonl', tail)
        write_jsonl(output / 'training.jsonl', log[:state['step']])
    return state


def train(config_path, data_root, output, resume=False):
    from run_reviewed_fitting import require_runner
    require_runner()
    cfg, rows, identity = fitting_identity(config_path, data_root)
    identity_hash = object_hash(identity)
    output = Path(output)
    require(resume or not output.exists(), 'Use a fresh fitting run or an unchanged checkpoint resume.')
    state = prepare_resume(output, identity_hash, cfg['steps']) if resume else None
    if state and state['step'] == cfg['steps']:
        return write_completion(output, cfg, state)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg['seed'])
    backend = backend_from(cfg)
    backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
    from satground.losses import Objective
    objective = Objective(cfg['region_weight'], cfg['boundary_weight'], cfg['perceptual_weight'],
                          cfg['training_segmenter_revision'], 'legacy')
    optimizer = torch.optim.AdamW(backend.adapter.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    start, prior_elapsed, prior_peak = 0, 0.0, 0.0
    if state:
        backend.adapter.load_state_dict(state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        start, prior_elapsed, prior_peak = state['step'], state['elapsed_seconds'], state['peak_vram_mib']
    else:
        save_json(output / 'run.json', dict(identity=identity, identity_hash=identity_hash, provenance=provenance(),
                  study='reviewed_training_fitting_diagnosis', main_study_eligible=False,
                  annotation_source='human_reviewed_corrected_training_proposals'))
        seed_everything(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    for step in range(start + 1, cfg['steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        parts, sampled = [], []
        for _ in range(cfg['accumulation']):
            row = rows[random.randrange(len(rows))]
            sampled.append(row['sample_id'])  # IDs remain in ignored local logs only.
            raw = backend.cached_features(safe_data_path(data_root, row['satellite']), cfg['feature_cache'])
            with Image.open(safe_data_path(data_root, row['panorama'])) as image:
                target = crop_panorama(image.convert('RGB'), row['yaw'], row['pitch'], row['fov'], cfg['size'])
            with np.load(Path(cfg['fitting_bundle']) / 'labels' / f"{row['sample_id']}.npz", allow_pickle=False) as values:
                labels = {k: torch.from_numpy(values[k]).cuda()[None, None] for k in ('building', 'valid')}
            prediction = backend.render(raw, row, adapted=True)
            loss, values = objective(prediction, rgb_tensor(target, 'cuda'), labels)
            require(bool(torch.isfinite(loss)) and finite_tree(values), f'Nonfinite fitting loss at step {step}.')
            scaler.scale(loss / cfg['accumulation']).backward()
            parts.append(values)
            del raw, target, labels, prediction, loss
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(backend.adapter.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        backend.assert_frozen()
        row = dict(step=step, gradient_norm=float(norm), elapsed_seconds=prior_elapsed + time.perf_counter() - begin,
                   peak_vram_mib=max(prior_peak, torch.cuda.max_memory_allocated() / 2**20),
                   sampled_view_ids=sampled, **{k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
        require(finite_tree(row), 'Nonfinite fitting update.')
        with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if step % cfg['save_every'] == 0 or step == cfg['steps']:
            check_pins(identity['source_file_sha256'])
            require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source changed during fitting.')
            state = dict(adapter=backend.adapter.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         rng=rng_state(), step=step, elapsed_seconds=row['elapsed_seconds'], peak_vram_mib=row['peak_vram_mib'],
                         identity_hash=identity_hash, config=cfg, provenance=provenance(),
                         train_manifest_sha256=identity['parent_training_manifest_sha256'],
                         fitting_manifest_sha256=identity['fitting_manifest_sha256'],
                         train_data_digest=identity['data_digest'], readiness=identity['readiness'],
                         trainable_parameters=sum(p.numel() for p in backend.adapter.parameters()))
            require(finite_tree(state), 'Nonfinite checkpoint; earlier outputs retained.')
            atomic_checkpoint(output / 'last.pt', state)
            periodic = output / f'step-{step:06d}.pt'
            temporary = periodic.with_suffix('.tmp')
            shutil.copyfile(output / 'last.pt', temporary)
            os.replace(temporary, periodic)
        if step == start + 1 or step % 10 == 0:
            print(json.dumps({k: v for k, v in row.items() if k != 'sampled_view_ids'}), flush=True)
    verify_bundle(cfg['fitting_bundle'])
    return write_completion(output, cfg, state)


def write_completion(output, cfg, state):
    require(state['step'] == cfg['steps'], 'Only final fixed-budget completion is reportable.')
    summary = dict(status='budget_complete', step=state['step'], identity_hash=state['identity_hash'],
                   elapsed_seconds=state['elapsed_seconds'], peak_vram_mib=state['peak_vram_mib'],
                   trainable_parameters=state['trainable_parameters'], checkpoint_sha256=sha256(Path(output) / 'last.pt'),
                   scope='training_fitting_only', main_study_eligible=False)
    save_json(Path(output) / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='data/vigor')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args.config, args.data_root, args.output, args.resume)
