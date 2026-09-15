"""Intervene on a completed adapter's fields on the fixed development views.

Writes all four predictions, raw opacity/radial maps, and per-view effects.
Run the ordinary independent evaluator on each prediction directory afterwards.
No target image is loaded by this generation script.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from satground.backend import load_style
from satground.common import (ResearchError, load_json, object_hash, provenance, read_jsonl,
                              require_development, safe_data_path, save_json, seed_everything,
                              sha256, write_jsonl)
from satground.fields import FIELD_MODES, render_fields
from satground.runs import backend_from, read_config


def diagnose(config_path, run_directory, manifest, data_root, output):
    cfg = read_config(config_path)
    if cfg['experiment'] not in ('A1', 'A3') or cfg.get('field_mode'):
        raise ResearchError('This diagnostic examines original A1/A3 adapters.')
    rows = read_jsonl(manifest)
    require_development(rows, {'validation'})
    if any(r['size'] != cfg['size'] for r in rows) or sha256(manifest) != sha256(cfg['validation_manifest']):
        raise ResearchError('Use the complete fixed validation manifest at its original resolution.')
    run = Path(run_directory)
    summary = load_json(run / 'summary.json')
    ckpt = run / 'last.pt'
    if summary['status'] != 'budget_complete' or summary['checkpoint_sha256'] != sha256(ckpt):
        raise ResearchError('Diagnostic requires a completed, checksum-verified adapter run.')
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if state['config'] != cfg or state['train_manifest_sha256'] != sha256(cfg['train_manifest']):
        raise ResearchError('Diagnostic config or training manifest differs from the checkpoint.')
    identity = load_json(run / 'run.json')['identity']
    if object_hash(identity) != state['identity_hash'] or identity['style_sha256'] != sha256(cfg['style']):
        raise ResearchError('Training provenance or fixed style has changed.')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ResearchError('Choose a fresh output directory; intervention records are immutable.')
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'protocol.json', dict(config=cfg, manifest_sha256=sha256(manifest),
        checkpoint_sha256=sha256(ckpt), sampling='frozen_A0_coarse_plus_importance',
        modes=FIELD_MODES, provenance=provenance(), target_conditioning=False,
        interpretation='field intervention; radial units are model coordinates, not measured geometry',
        human_review_pending=True))
    backend = backend_from(cfg)
    backend.adapter.load_state_dict(state['adapter'])
    backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
    predictions = {m: [] for m in FIELD_MODES}
    for mode in FIELD_MODES:
        (out / mode).mkdir()
    (out / 'geometry').mkdir()
    effects = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(rows):
        raw = backend.cached_features(safe_data_path(data_root, row['satellite']), cfg['feature_cache'])
        render_seed = int(object_hash([cfg['render_seed'], row['sample_id']])[:8], 16)
        with torch.no_grad():
            seed_everything(render_seed)
            result = render_fields(backend, raw, row)
            seed_everything(render_seed)
            native_base = backend.render(raw, row)
            seed_everything(render_seed)
            native_joint = backend.render(raw, row, adapted=True)
        base_error = float((native_base - result['base']['rgb']).abs().max())
        if base_error > 3e-5:
            raise ResearchError('Shared baseline differs from the native chunked baseline.')
        for key in ('opacity', 'radial'):
            if not torch.equal(result['appearance'][key], result['base'][key]):
                raise ResearchError('Appearance intervention changed geometry.')
            if not torch.equal(result['density'][key], result['joint'][key]):
                raise ResearchError('Density intervention differs from joint geometry.')
        record = dict(sample_id=row['sample_id'], geo_group=row['geo_group'], split=row['split'],
            native_baseline_max_rgb_difference=base_error,
            native_vs_shared_joint_rgb_mae=float((native_joint - result['joint']['rgb']).abs().mean()))
        arrays = {}
        for mode in FIELD_MODES:
            image = (result[mode]['rgb'][0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            name = f"{row['sample_id']}.png"
            Image.fromarray(image).save(out / mode / name)
            predictions[mode].append(dict(sample_id=row['sample_id'], prediction=name,
                sha256=sha256(out / mode / name),
                input_camera={k: row[k] for k in ('h_offset', 'w_offset', 'yaw', 'pitch', 'fov', 'size')}))
            for key in ('opacity', 'radial'):
                arrays[f'{mode}_{key}'] = result[mode][key][0, 0].cpu().numpy()
                record[f'{mode}_{key}_mae_from_base'] = float((result[mode][key] - result['base'][key]).abs().mean())
            record[f'{mode}_rgb_mae_from_base'] = float((result[mode]['rgb'] - result['base']['rgb']).abs().mean())
        path = out / 'geometry' / f"{row['sample_id']}.npz"
        np.savez_compressed(path, **arrays)
        record['geometry_sha256'] = sha256(path)
        effects.append(record)
        print(f"{cfg['experiment']}: intervention {i + 1}/{len(rows)}, baseline max difference {base_error:.2g}", flush=True)
        del result, raw, native_base, native_joint
    write_jsonl(out / 'effects.jsonl', effects)
    generation = dict(config=cfg, manifest_sha256=sha256(manifest), style_sha256=sha256(cfg['style']),
        checkpoint_sha256=sha256(ckpt), provenance=provenance(), stress='clean',
        peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20,
        elapsed_seconds=time.perf_counter() - begin, count=len(rows),
        trained_steps=state['step'], training_identity_hash=state['identity_hash'],
        training_provenance=state['provenance'], train_manifest_sha256=state['train_manifest_sha256'],
        train_data_digest=state['train_data_digest'], target_images_used_for_conditioning=False,
        scientific=True, sampling='frozen_A0_coarse_plus_importance',
        scientific_improvement_established=False, human_review_pending=True)
    for mode in FIELD_MODES:
        write_jsonl(out / mode / 'predictions.jsonl', predictions[mode])
        save_json(out / mode / 'generation.json', dict(generation, field_intervention=mode))
    save_json(out / 'summary.json', dict(generation, status='complete', effects_sha256=sha256(out / 'effects.jsonl'),
        native_baseline_max_rgb_difference=max(r['native_baseline_max_rgb_difference'] for r in effects),
        geometry_control_checks_passed=True))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'run', 'manifest', 'data-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(diagnose(args.config, args.run, args.manifest, args.data_root, args.output))
