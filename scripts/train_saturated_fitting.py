"""One matched fixed-budget fitting pair with unchanged masks and saturated v2."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from audit_reviewed_fitting_outputs import checkpoint_payload
from prepare_saturated_fitting import make_configs, source_pins, verify_trial
from prepare_reviewed_fitting import check_pins, require
from saturated_surface_objective import prepare_surface_labels, valid_surface_loss
from satground.adapter import StructuralAdapter
from satground.backend import load_style
from satground.camera import crop_panorama
from satground.common import (ROOT, load_json, object_hash, provenance, read_jsonl, safe_data_path,
    save_json, seed_everything, sha256, write_jsonl)
from satground.losses import Objective, structural_losses
from satground.readiness import require_training_readiness
from satground.runs import atomic_checkpoint, backend_from, read_config, restore_rng, rng_state
from satground.semantics import rgb_tensor
from train_reviewed_fitting import finite_tree, write_completion


class SaturatedObjective:
    """Same frozen modules and one segmenter forward for both A2 and A3."""
    def __init__(self, cfg, reference=None):
        self.reference = reference or Objective(cfg['region_weight'], 0., cfg['perceptual_weight'],
                                                cfg['training_segmenter_revision'], 'legacy')
        self.perceptual, self.segmenter = self.reference.perceptual, self.reference.segmenter
        self.cfg = cfg

    def __call__(self, prediction, target, labels, prepared):
        probability = self.segmenter.probabilities(prediction)[:, 2:3]
        region, diagnostic = structural_losses(probability, labels['building'], labels['valid'])
        rgb = F.l1_loss(prediction, target)
        lpips = self.perceptual(prediction * 2 - 1, target * 2 - 1).mean()
        surface = valid_surface_loss(probability, prepared)
        total = rgb + self.cfg['perceptual_weight'] * lpips + self.cfg['region_weight'] * region
        total = total + self.cfg['surface_weight'] * surface
        parts = dict(rgb=rgb, lpips=lpips, region=region, legacy_boundary_diagnostic=diagnostic,
                     surface=surface, total=total)
        values = {k: float(v.detach()) for k, v in parts.items()}
        values.update(boundary=0., boundary_weight=0., surface_weight=self.cfg['surface_weight'])
        return total, values


def fitting_identity(config_path, data_root):
    cfg = read_config(config_path)
    require(cfg['experiment'] in ('A2', 'A3'), 'Only fresh A2/A3 receive saturated fitting updates.')
    trial = Path(cfg['saturated_trial']).resolve()
    record = verify_trial(trial)
    original = Path(record['original_bundle'])
    require(cfg == make_configs(trial, record['protocol'], original)[cfg['experiment']]
            and str(Path(config_path).resolve()) in record['artifact_sha256'], 'Configuration is not the frozen trial config.')
    rows = read_jsonl(cfg['fitting_manifest'])
    require(len(rows) == 8, 'Only the original eight reviewed training views are allowed.')
    readiness = require_training_readiness(cfg, rows, cfg['steps'])
    verified = {r['path']: r['sha256'] for r in load_json(ROOT / 'reports/data-readiness.json')['files']}
    hashes = {}
    for row in rows:
        for key in ('satellite', 'panorama'):
            path = safe_data_path(data_root, row[key])
            digest = sha256(path)
            require(verified.get(row[key]) == digest and record['source_file_sha256'].get(str(path)) == digest,
                    'Training RGB differs from the frozen verification.')
            hashes[row[key]] = digest
    p = provenance()
    identity = dict(config=cfg, trial_sha256=sha256(trial / 'trial.json'),
        bundle_sha256=record['original_bundle_sha256'], bundle_inputs=record['source_file_sha256'],
        bundle_artifacts=record['artifact_sha256'], source_file_sha256=source_pins(),
        pipeline_source_hash=p['pipeline_source_hash'],
        environment={**{k: p[k] for k in ('python', 'packages', 'code_revision', 'model_revision')},
            'torch_cpu_threads': torch.get_num_threads(), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32}, readiness=readiness, data_digest=object_hash(hashes),
        parent_training_manifest_sha256=sha256(cfg['train_manifest']),
        fitting_manifest_sha256=sha256(cfg['fitting_manifest']), style_sha256=sha256(cfg['style']))
    return cfg, rows, identity


def verify_sample_prefix(log, rows, cfg, python_rng):
    sampler = random.Random(cfg['seed'])
    expected = [[rows[sampler.randrange(len(rows))]['sample_id'] for _ in range(cfg['accumulation'])] for _ in log]
    require([r['sampled_view_ids'] for r in log] == expected and sampler.getstate() == python_rng,
            'Training sample prefix or saved Python sampler RNG differs.')


def prepare_resume(output, identity_hash, budget, rows):
    output = Path(output)
    state = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    run = load_json(output / 'run.json')
    require(state['identity_hash'] == identity_hash == run['identity_hash'] == object_hash(run['identity'])
            and state['config'] == run['identity']['config'], 'Cannot resume altered saturated fitting inputs or source.')
    require(type(state['step']) is int and 0 < state['step'] <= budget, 'Invalid recovery update count.')
    shapes = {k: tuple(v.shape) for k, v in StructuralAdapter(64, 16).state_dict().items()}
    checkpoint_payload(state, shapes, state['step'], state['config'])
    log = read_jsonl(output / 'training.jsonl')
    require([r['step'] for r in log] == list(range(1, len(log) + 1)) and state['step'] <= len(log) <= budget
            and finite_tree(log), 'Training log does not support the saturated recovery checkpoint.')
    verify_sample_prefix(log[:state['step']], rows, state['config'], state['rng']['python'])
    receipt_path = output / 'startup-receipt.json'
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt['step'] == 1 and receipt['identity_hash'] == identity_hash
                and receipt['first_log_row_sha256'] == object_hash(log[0]), 'Startup receipt differs from its exact prefix.')
        startup = output / 'startup.pt'
        require(startup.is_file() and receipt['startup_checkpoint_sha256'] == receipt['checkpoint_sha256'] == sha256(startup),
                'Retained startup checkpoint bytes differ from the receipt.')
        first = torch.load(startup, map_location='cpu', weights_only=True)
        require(first['step'] == 1 and first['identity_hash'] == identity_hash and first['config'] == state['config'],
                'Retained startup checkpoint identity differs.')
        checkpoint_payload(first, shapes, 1, state['config'])
        verify_sample_prefix(log[:1], rows, state['config'], first['rng']['python'])
        if state['step'] == 1:
            require(receipt['checkpoint_sha256'] == sha256(output / 'last.pt'), 'Startup checkpoint bytes changed.')
    tail = log[state['step']:]
    if tail:
        write_jsonl(output / f'interrupted-steps-{uuid4().hex}.jsonl', tail)
        write_jsonl(output / 'training.jsonl', log[:state['step']])
    return state


def save_checkpoint(output, backend, optimizer, scaler, cfg, identity, identity_hash, row, periodic):
    check_pins(identity['source_file_sha256'])
    require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source changed during fitting.')
    state = dict(adapter=backend.adapter.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        rng=rng_state(), step=row['step'], elapsed_seconds=row['elapsed_seconds'], peak_vram_mib=row['peak_vram_mib'],
        identity_hash=identity_hash, config=cfg, provenance=provenance(),
        train_manifest_sha256=identity['parent_training_manifest_sha256'],
        fitting_manifest_sha256=identity['fitting_manifest_sha256'], train_data_digest=identity['data_digest'],
        readiness=identity['readiness'], trainable_parameters=sum(p.numel() for p in backend.adapter.parameters()))
    require(finite_tree(state), 'Nonfinite checkpoint; partial outputs retained.')
    atomic_checkpoint(output / 'last.pt', state)
    if periodic:
        target = output / f"step-{row['step']:06d}.pt"
        temporary = target.with_suffix('.tmp')
        shutil.copyfile(output / 'last.pt', temporary)
        os.replace(temporary, target)
    return state


def save_resume_verification(output, identity_hash, step, budget, restored):
    output = Path(output)
    path = output / 'resume-verification.json'
    if path.exists():
        # Retain earlier recovery evidence rather than silently overwriting it.
        shutil.copyfile(path, output / f'resume-verification-{uuid4().hex}.json')
    proof = dict(identity_hash=identity_hash, restored_step=step, remaining_updates=budget - step,
        checkpoint_sha256=sha256(output / 'last.pt'), sample_prefix_verified=True,
        checkpoint_payload_verified=True, adapter_optimizer_scaler_rng_restored=restored,
        exact_adapter_optimizer_scaler_readback_verified=restored,
        exact_python_numpy_torch_cpu_cuda_rng_readback_verified=restored,
        completion_without_optimizer_updates=not restored, optimizer_updates_performed=0)
    save_json(path, proof)
    return proof


def exact_payload_equal(actual, expected):
    """Device-independent exact state comparison, with no draws or updates."""
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
                and actual.dtype == expected.dtype and actual.shape == expected.shape
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys()
                and all(exact_payload_equal(actual[k], expected[k]) for k in actual))
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        return (type(actual) is type(expected) and len(actual) == len(expected)
                and all(exact_payload_equal(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def train(config_path, data_root, output, resume=False, startup_one_update=False):
    from run_saturated_fitting import require_runner
    require_runner()
    cfg, rows, identity = fitting_identity(config_path, data_root)
    require(not (resume and startup_one_update) and (not startup_one_update or cfg['experiment'] == 'A2'),
            'The sole startup update is fresh A2 step1, included in the budget.')
    identity_hash, output = object_hash(identity), Path(output).resolve()
    require(output.is_relative_to(ROOT / 'runs') and (resume or not output.exists()), 'Use fresh private output or compatible resume.')
    state = prepare_resume(output, identity_hash, cfg['steps'], rows) if resume else None
    if state and state['step'] == cfg['steps']:
        save_resume_verification(output, identity_hash, state['step'], cfg['steps'], restored=False)
        return write_completion(output, cfg, state)
    prepared = {}
    for row in rows:
        with np.load(Path(cfg['fitting_bundle']) / 'labels' / (row['sample_id'] + '.npz'), allow_pickle=False) as values:
            prepared[row['sample_id']] = prepare_surface_labels(values['building'], values['valid'])
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg['seed'])
    backend = backend_from(cfg)
    backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
    objective = SaturatedObjective(cfg)
    optimizer = torch.optim.AdamW(backend.adapter.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    start, elapsed, peak = 0, 0., 0.
    if state:
        backend.adapter.load_state_dict(state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])  # All allocations/module loading occur before restoring the saved stream.
        require(exact_payload_equal(backend.adapter.state_dict(), state['adapter'])
                and exact_payload_equal(optimizer.state_dict(), state['optimizer'])
                and exact_payload_equal(scaler.state_dict(), state['scaler'])
                and exact_payload_equal(rng_state(), state['rng']),
                'Actual adapter/optimizer/scaler/RNG readback differs immediately after restoration.')
        start, elapsed, peak = state['step'], state['elapsed_seconds'], state['peak_vram_mib']
        save_resume_verification(output, identity_hash, start, cfg['steps'], restored=True)
    else:
        save_json(output / 'run.json', dict(identity=identity, identity_hash=identity_hash, provenance=provenance(),
            study='saturated_reviewed_training_fitting', main_study_eligible=False,
            annotation_source='unchanged_original_eight_reviewed_training_masks'))
        seed_everything(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    end = 1 if startup_one_update else cfg['steps']
    for step in range(start + 1, end + 1):
        optimizer.zero_grad(set_to_none=True)
        parts, sampled = [], []
        for _ in range(cfg['accumulation']):
            row = rows[random.randrange(len(rows))]
            sampled.append(row['sample_id'])
            raw = backend.cached_features(safe_data_path(data_root, row['satellite']), cfg['feature_cache'])
            with Image.open(safe_data_path(data_root, row['panorama'])) as image:
                target = crop_panorama(image.convert('RGB'), row['yaw'], row['pitch'], row['fov'], cfg['size'])
            with np.load(Path(cfg['fitting_bundle']) / 'labels' / (row['sample_id'] + '.npz'), allow_pickle=False) as values:
                labels = {k: torch.from_numpy(values[k]).cuda()[None, None] for k in ('building', 'valid')}
            prediction = backend.render(raw, row, adapted=True)
            loss, values = objective(prediction, rgb_tensor(target, 'cuda'), labels, prepared[row['sample_id']])
            require(bool(torch.isfinite(loss)) and finite_tree(values), f'Nonfinite saturated loss at step{step}.')
            scaler.scale(loss / cfg['accumulation']).backward()
            parts.append(values)
            del raw, target, labels, prediction, loss
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(backend.adapter.parameters(), 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        backend.assert_frozen()
        require(all(not p.requires_grad and p.grad is None for module in (objective.perceptual, objective.segmenter.model)
                    for p in module.parameters()), 'Reference modules are not frozen.')
        row = dict(step=step, gradient_norm=float(norm), elapsed_seconds=elapsed + time.perf_counter() - begin,
            peak_vram_mib=max(peak, torch.cuda.max_memory_allocated() / 2**20), sampled_view_ids=sampled,
            **{k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
        require(finite_tree(row), 'Nonfinite saturated update.')
        with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        periodic = step % cfg['save_every'] == 0 or step == cfg['steps']
        if periodic or startup_one_update:
            state = save_checkpoint(output, backend, optimizer, scaler, cfg, identity, identity_hash, row, periodic)
        if step == start + 1 or step % 10 == 0:
            print(json.dumps({k: v for k, v in row.items() if k != 'sampled_view_ids'}), flush=True)
        if startup_one_update:
            verify_sample_prefix([row], rows, cfg, state['rng']['python'])
            temporary = output / 'startup.tmp'
            shutil.copyfile(output / 'last.pt', temporary)
            os.replace(temporary, output / 'startup.pt')
            receipt = dict(status='startup_complete', step=1, identity_hash=identity_hash,
                checkpoint_sha256=sha256(output / 'last.pt'), startup_checkpoint_sha256=sha256(output / 'startup.pt'),
                first_log_row_sha256=object_hash(row),
                budget_steps=200, remaining_updates=199, updates_included_in_budget=True,
                sample_prefix_verified=True)
            save_json(output / 'startup-receipt.json', receipt)
            return receipt
    verify_trial(cfg['saturated_trial'])
    return write_completion(output, cfg, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='data/vigor')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--startup-one-update', action='store_true')
    args = parser.parse_args()
    train(args.config, args.data_root, args.output, args.resume, args.startup_one_update)
