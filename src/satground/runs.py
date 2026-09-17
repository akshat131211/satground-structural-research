"""Experiment lifecycle, resumable adaptation, fixed-pose generation, and profiling."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .backend import Sat3DBackend, load_style
from .camera import crop_panorama, perturb_position_width
from .common import (ROOT, ResearchError, load_json, object_hash, provenance, read_jsonl,
                     require_development, safe_data_path, save_json, seed_everything, sha256, write_jsonl)
from .manifests import assert_disjoint, require_files
from .semantics import rgb_tensor


def read_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['experiment'] not in ('A0', 'A1', 'A2', 'A3', 'GJ', 'GD'):
        raise ResearchError('Unknown experiment. A4 operates on the trained A3 ensemble.')
    expected_mode = {'GJ': 'joint', 'GD': 'density'}.get(cfg['experiment'])
    if cfg.get('field_mode') != expected_mode:
        raise ResearchError('GJ/GD require their explicit shared-sampling field mode; original A experiments must retain their renderer.')
    if cfg['experiment'] in ('A0', 'A1') and (cfg['region_weight'] or cfg['boundary_weight']):
        raise ResearchError('A0/A1 must not use structural losses.')
    if cfg['experiment'] == 'A2' and cfg['boundary_weight']:
        raise ResearchError('A2 must not use boundary supervision.')
    if cfg.get('boundary_mode', 'legacy') not in ('legacy', 'anchored_contour_v1'):
        raise ResearchError('Unknown boundary target mode.')
    if cfg.get('boundary_mode') == 'anchored_contour_v1' and cfg['experiment'] not in ('A2', 'A3'):
        raise ResearchError('The contour pilot supports only matched A2/A3 adaptation.')
    if cfg['steps'] < 1 or cfg['accumulation'] < 1 or cfg['save_every'] < 1 or cfg['learning_rate'] <= 0:
        raise ResearchError('Training budget, accumulation, checkpoint interval, and learning rate must be positive.')
    return cfg


def backend_from(cfg):
    return Sat3DBackend(size=cfg['size'], chunk_rows=cfg['chunk_rows'],
                        checkpoint_rays=cfg['checkpoint_rays'], amp=cfg['amp'], width=cfg['adapter_width'],
                        field_mode=cfg.get('field_mode'))


def rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(temp, path)


def validate_labels(directory, manifest, data_root):
    record = load_json(Path(directory) / 'provenance.json')
    if record.get('kind') != 'pseudo_labels' or record['manifest_sha256'] != sha256(manifest):
        raise ResearchError('Pseudo-label provenance does not match the training manifest.')
    for path, digest in record['target_sha256'].items():
        if sha256(safe_data_path(data_root, path)) != digest:
            raise ResearchError('A pseudo-label target image has changed.')
    for sample_id, digest in record['label_sha256'].items():
        if sha256(Path(directory) / f'{sample_id}.npz') != digest:
            raise ResearchError('A cached pseudo-label has changed.')
    return record


def train(config_path, data_root, output, seed=None, resume=False, stop_after=None):
    cfg = read_config(config_path)
    if cfg['experiment'] == 'A0':
        raise ResearchError('A0 is frozen; use generate.')
    if seed is not None:
        cfg['seed'] = seed
    rows = read_jsonl(cfg['train_manifest'])
    validation = read_jsonl(cfg['validation_manifest'])
    require_development(rows, {'train'})
    require_development(validation, {'validation'})
    assert_disjoint(rows + validation)
    require_files(rows + validation, data_root)
    from .readiness import require_training_readiness
    readiness = require_training_readiness(cfg, rows + validation, min(cfg['steps'], stop_after) if stop_after else cfg['steps'])
    if any(r['size'] != cfg['size'] for r in rows + validation):
        raise ResearchError('Manifest crop size differs from experiment resolution.')
    style = load_style(cfg['style'], sha256(cfg['train_manifest']))
    label_record = validate_labels(cfg['labels'], cfg['train_manifest'], data_root) if cfg['region_weight'] or cfg['boundary_weight'] else None
    if cfg.get('boundary_mode') == 'anchored_contour_v1':
        from .contours import validate_contour_labels
        validate_contour_labels(label_record, cfg['labels'], rows)
    image_paths = sorted({r[k] for r in rows for k in ('satellite', 'panorama')})
    image_hashes = {p: sha256(safe_data_path(data_root, p)) for p in image_paths}
    verified = {r['path']: r['sha256'] for r in load_json(cfg.get('data_validation', 'reports/data-readiness.json'))['files']}
    if any(verified.get(p) != digest for p, digest in image_hashes.items()):
        raise ResearchError('Training images changed after RGB validation; repeat data QA.')
    data_digest = object_hash(image_hashes)
    identity = dict(config=cfg, train_manifest_sha256=sha256(cfg['train_manifest']),
                    validation_manifest_sha256=sha256(cfg['validation_manifest']),
                    style_sha256=sha256(cfg['style']), data_digest=data_digest, label_record=label_record,
                    pipeline_source_hash=provenance()['pipeline_source_hash'], readiness=readiness)
    identity_hash = object_hash(identity)
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not resume:
        raise ResearchError('Run directory already has outputs; use --resume or a fresh directory.')
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg['seed'])
    backend = backend_from(cfg)
    backend.set_style(style)
    from .losses import Objective
    objective = Objective(cfg['region_weight'], cfg['boundary_weight'], cfg['perceptual_weight'],
                          label_record['revision'] if label_record else None, cfg.get('boundary_mode', 'legacy'))
    optimizer = torch.optim.AdamW(backend.adapter.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    start, prior_elapsed = 0, 0.0
    checkpoint_path = out / 'last.pt'
    if resume:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if state['identity_hash'] != identity_hash:
            raise ResearchError('Cannot resume: configuration, data, labels, or style changed.')
        backend.adapter.load_state_dict(state['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        start, prior_elapsed = state['step'], state['elapsed_seconds']
        log_path = out / 'training.jsonl'
        if log_path.exists():
            log = read_jsonl(log_path)
            rolled_back = [r for r in log if r['step'] > start]
            if rolled_back:
                from uuid import uuid4
                write_jsonl(out / f'interrupted-steps-{uuid4().hex}.jsonl', rolled_back)
                write_jsonl(log_path, [r for r in log if r['step'] <= start])
    else:
        save_json(out / 'run.json', dict(provenance=provenance(), identity=identity, identity_hash=identity_hash,
                                        status='running', scientific=True, human_review_pending=True,
                                        readiness=readiness))
    # Model/metric initialization may consume randomness differently in A1 vs A3.
    if not resume:
        seed_everything(cfg['seed'])
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    end = min(cfg['steps'], stop_after) if stop_after else cfg['steps']
    if end <= start:
        raise ResearchError('Requested stopping step is not later than the saved step.')
    for step in range(start, end):
        optimizer.zero_grad(set_to_none=True)
        parts = []
        for micro in range(cfg['accumulation']):
            row = rows[random.randrange(len(rows))]
            raw = backend.cached_features(safe_data_path(data_root, row['satellite']), cfg['feature_cache'])
            target = crop_panorama(Image.open(safe_data_path(data_root, row['panorama'])).convert('RGB'),
                                   row['yaw'], row['pitch'], row['fov'], cfg['size'])
            labels = None
            if label_record:
                with np.load(Path(cfg['labels']) / f"{row['sample_id']}.npz") as data:
                    keys = ('building', 'valid', 'boundary_valid') if cfg.get('boundary_mode') == 'anchored_contour_v1' else ('building', 'valid')
                    labels = {k: torch.from_numpy(data[k].astype(np.float32)).cuda()[None, None] for k in keys}
            prediction = backend.render(raw, row, adapted=True)
            # Loss/semantic operations remain float32 for stable probability and boundary gradients.
            loss, values = objective(prediction, rgb_tensor(target, 'cuda'), labels)
            if not torch.isfinite(loss):
                raise ResearchError(f'Non-finite loss at optimizer step {step + 1}.')
            scaler.scale(loss / cfg['accumulation']).backward()
            parts.append(values)
            del prediction, raw, loss, target, labels
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(backend.adapter.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        backend.assert_frozen()
        record = dict(step=step + 1, gradient_norm=float(gradient_norm),
                      elapsed_seconds=prior_elapsed + time.perf_counter() - begin,
                      peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                      **{k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
        with (out / 'training.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if (step + 1) % cfg['save_every'] == 0 or step + 1 == end:
            state = dict(adapter=backend.adapter.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         rng=rng_state(), step=step + 1, elapsed_seconds=record['elapsed_seconds'], identity_hash=identity_hash,
                         config=cfg, provenance=provenance(), train_manifest_sha256=sha256(cfg['train_manifest']),
                         train_data_digest=data_digest, readiness=readiness)
            atomic_checkpoint(checkpoint_path, state)
            atomic_checkpoint(out / f'step-{step + 1:06d}.pt', state)
        if (step + 1) % 10 == 0 or step == start:
            print(json.dumps(record), flush=True)
    summary = dict(status='budget_complete' if end == cfg['steps'] else 'paused_at_requested_step', step=end,
                    peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                    elapsed_seconds=prior_elapsed + time.perf_counter() - begin,
                    trainable_parameters=sum(p.numel() for p in backend.adapter.parameters()),
                    checkpoint_sha256=sha256(checkpoint_path), validation_selection_pending=True,
                    readiness=readiness)
    save_json(out / 'summary.json', summary)
    return summary


def freeze_audit(selection_path, audit_manifest, output):
    selection = load_json(selection_path)
    rows = read_jsonl(audit_manifest)
    require_development(rows, {'audit'})
    if Path(output).exists():
        raise ResearchError('Audit freeze already exists; it is immutable.')
    if not selection.get('validation_selection_complete') or not selection.get('runs'):
        raise ResearchError('Record completed validation selection and selected run files before freezing the audit.')
    for run in selection['runs']:
        run['config_sha256'] = sha256(run['config'])
        run['checkpoint_sha256'] = sha256(run['checkpoint']) if run.get('checkpoint') else None
    record = dict(selection=selection, audit_manifest_sha256=sha256(audit_manifest), created=provenance())
    save_json(output, record)
    return record


def generate(config_path, manifest, data_root, output, checkpoint_path=None, audit_freeze=None, stress='clean'):
    cfg = read_config(config_path)
    rows = read_jsonl(manifest)
    require_development(rows)
    if len({r['split'] for r in rows}) != 1:
        raise ResearchError('Generate one split at a time.')
    if any(r['size'] != cfg['size'] for r in rows):
        raise ResearchError('Manifest/config resolution mismatch.')
    if cfg['experiment'] != 'A0' and not checkpoint_path:
        raise ResearchError('Adapted generation requires a trained checkpoint.')
    if cfg['experiment'] == 'A0' and checkpoint_path:
        raise ResearchError('A0 cannot use an adapter checkpoint.')
    style = load_style(cfg['style'], sha256(cfg['train_manifest']))
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ResearchError('Prediction directory already exists. Do not overwrite fixed evaluation outputs.')
    checkpoint_hash = sha256(checkpoint_path) if checkpoint_path else None
    if rows[0]['split'] == 'audit':
        if cfg.get('data_review_mode') == 'exploratory':
            raise ResearchError('Exploratory runs with pending data review cannot open the sealed audit.')
        if not audit_freeze:
            raise ResearchError('Audit generation requires an immutable validation selection freeze.')
        frozen = load_json(audit_freeze)
        if frozen['audit_manifest_sha256'] != sha256(manifest):
            raise ResearchError('Audit manifest differs from frozen selection.')
        if not any(r['config_sha256'] == sha256(config_path) and r['checkpoint_sha256'] == checkpoint_hash for r in frozen['selection']['runs']):
            raise ResearchError('This run was not selected before audit evaluation.')
        ledger = Path(audit_freeze).with_suffix('.ledger')
        ledger.mkdir(exist_ok=True)
        key = object_hash([sha256(config_path), checkpoint_hash, stress])
        with (ledger / key).open('x') as stream:
            stream.write(str(out.resolve()))
    seed_everything(cfg['seed'])
    backend = backend_from(cfg)
    backend.set_style(style)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if {k: v for k, v in state['config'].items() if k != 'seed'} != {k: v for k, v in cfg.items() if k != 'seed'}:
            raise ResearchError('Checkpoint configuration differs from generation configuration.')
        cfg['seed'] = state['config']['seed']
        backend.adapter.load_state_dict(state['adapter'])
    out.mkdir(parents=True, exist_ok=True)
    predictions = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        seed_everything(int(object_hash([cfg['render_seed'], row['sample_id']])[:8], 16))
        path = safe_data_path(data_root, row['satellite'])
        camera = dict(row)
        if stress == 'clean' or stress in ('yaw_error', 'position_error'):
            raw = backend.cached_features(path, cfg['feature_cache'])
        else:
            from PIL import ImageFilter
            image = Image.open(path).convert('RGB').resize((256, 256), Image.Resampling.BICUBIC)
            if stress == 'resolution':
                image = image.resize((64, 64), Image.Resampling.BILINEAR).resize((256, 256), Image.Resampling.BILINEAR)
            elif stress == 'blur':
                image = image.filter(ImageFilter.GaussianBlur(2))
            elif stress == 'occlusion':
                image.paste((127, 127, 127), (96, 96, 160, 160))
            else:
                raise ResearchError('Unknown stress test.')
            x = rgb_tensor(image)
            raw = backend.encode((x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None])
        if stress == 'yaw_error':
            camera['yaw'] += 5
        if stress == 'position_error':
            camera['w_offset'] = perturb_position_width(camera['w_offset'])
        with torch.no_grad():
            pred = backend.render(raw, camera, adapted=bool(checkpoint_path))
        name = f"{row['sample_id']}.png"
        array = (pred[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        Image.fromarray(array).save(out / name)
        predictions.append(dict(sample_id=row['sample_id'], prediction=name, sha256=sha256(out / name),
                                input_camera={k: camera[k] for k in ('h_offset', 'w_offset', 'yaw', 'pitch', 'fov', 'size')}))
        del raw, pred
    write_jsonl(out / 'predictions.jsonl', predictions)
    record = dict(config=cfg, manifest_sha256=sha256(manifest), style_sha256=sha256(cfg['style']),
                  checkpoint_sha256=checkpoint_hash, provenance=provenance(), stress=stress,
                  peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                  elapsed_seconds=time.perf_counter() - begin, count=len(predictions),
                  trained_steps=state['step'] if checkpoint_path else 0,
                  training_identity_hash=state['identity_hash'] if checkpoint_path else None,
                  training_provenance=state['provenance'] if checkpoint_path else None,
                  training_readiness=state.get('readiness') if checkpoint_path else None,
                  train_manifest_sha256=state['train_manifest_sha256'] if checkpoint_path else sha256(cfg['train_manifest']),
                  train_data_digest=state['train_data_digest'] if checkpoint_path else None,
                  audit_freeze_sha256=sha256(audit_freeze) if audit_freeze else None,
                  target_images_used_for_conditioning=False, scientific=True)
    save_json(out / 'generation.json', record)
    return record


def profile(output, size=128, chunk_rows=4, steps=3, amp=False, full_loss=False, accumulation=1):
    """Synthetic inputs prove software/VRAM behavior only, never model quality."""
    if steps < 1 or accumulation < 1:
        raise ResearchError('Profile steps and accumulation must be positive.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(provenance=provenance(), synthetic=True, scientific_improvement_evidence=False,
                  size=size, chunk_rows=chunk_rows, steps_requested=steps, amp=amp, full_loss=full_loss,
                  accumulation=accumulation)
    start = time.perf_counter()
    try:
        seed_everything(17)
        torch.cuda.reset_peak_memory_stats()
        backend = Sat3DBackend(size, chunk_rows, True, amp)
        raw = backend.encode(torch.zeros(1, 3, 256, 256))
        backend.set_style(np.ones(270, np.float32) / 90)
        camera = dict(h_offset=0, w_offset=0, yaw=0, pitch=0, fov=90)
        with torch.no_grad():
            seed_everything(123)
            base = backend.render(raw, camera)
            seed_everything(123)
            initial = backend.render(raw, camera, adapted=True)
        record['zero_adapter_max_rgb_difference'] = float((base - initial).abs().max())
        if not torch.equal(base, initial):
            raise ResearchError('Zero adapter did not reproduce the frozen baseline exactly.')
        Image.fromarray((base[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)).save(out / 'SYNTHETIC_INPUT_baseline.png')
        from .losses import Objective
        objective = Objective(.5, .5, .1) if full_loss else None
        if objective:
            record['structural_segmenter'] = dict(model_id=objective.segmenter.model_id,
                                                  revision=objective.segmenter.revision,
                                                  input_size=objective.segmenter.input_size)
        target = torch.full_like(base, .5)
        labels = dict(building=torch.zeros_like(target[:, :1]), valid=torch.ones_like(target[:, :1]))
        optimizer = torch.optim.AdamW(backend.adapter.parameters(), lr=1e-4)
        losses, norms, durations = [], [], []
        for index in range(steps):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(accumulation):
                prediction = backend.render(raw, camera, adapted=True)
                loss = objective(prediction, target, labels)[0] if objective else (prediction - target).square().mean()
                if not torch.isfinite(loss):
                    raise ResearchError('Profile loss is not finite.')
                (loss / accumulation).backward()
                micro_losses.append(float(loss.detach()))
                del prediction, loss
            norm = torch.nn.utils.clip_grad_norm_(backend.adapter.parameters(), 1, error_if_nonfinite=True)
            if float(norm) == 0:
                raise ResearchError('Adapter received no gradient.')
            optimizer.step()
            backend.assert_frozen()
            torch.cuda.synchronize()
            losses.append(float(np.mean(micro_losses)))
            norms.append(float(norm))
            durations.append(time.perf_counter() - begin)
            print(json.dumps(dict(profile_step=index + 1, seconds=durations[-1], loss=losses[-1])), flush=True)
        # Exercise actual optimizer + adapter + CUDA RNG serialization before a follow-up step.
        saved = dict(adapter=backend.adapter.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
        atomic_checkpoint(out / 'synthetic-resume.pt', saved)
        reloaded = torch.load(out / 'synthetic-resume.pt', map_location='cpu', weights_only=True)
        backend.adapter.load_state_dict(reloaded['adapter'])
        optimizer.load_state_dict(reloaded['optimizer'])
        restore_rng(reloaded['rng'])
        optimizer.zero_grad(set_to_none=True)
        resumed_finite = True
        for _ in range(accumulation):
            resumed_prediction = backend.render(raw, camera, adapted=True)
            resumed_loss = objective(resumed_prediction, target, labels)[0] if objective else (resumed_prediction - target).square().mean()
            resumed_finite = resumed_finite and bool(torch.isfinite(resumed_loss))
            (resumed_loss / accumulation).backward()
            del resumed_prediction, resumed_loss
        torch.nn.utils.clip_grad_norm_(backend.adapter.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        backend.assert_frozen()
        record.update(status='passed', losses=losses, gradient_norms=norms, step_seconds=durations,
                      resume_step_finite=resumed_finite,
                      trainable_parameters=sum(p.numel() for p in backend.adapter.parameters()),
                      frozen_parameters=sum(p.numel() for p in backend.model.parameters()),
                      peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                      device=torch.cuda.get_device_name())
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_json(out / 'profile.json', dict(record, elapsed_seconds=time.perf_counter() - start))
        raise
    record['elapsed_seconds'] = time.perf_counter() - start
    save_json(out / 'profile.json', record)
    return record
