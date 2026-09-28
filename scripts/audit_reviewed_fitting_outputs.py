"""CPU-only completion checks supplementary to the unchanged fitting reporter."""
import argparse
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import psutil
import torch

from prepare_reviewed_fitting import check_pins, require, verify_bundle
from report_reviewed_fitting import training_audit
from run_reviewed_fitting import LOCK
from satground.adapter import StructuralAdapter
from satground.annotations import binary_mask
from satground.common import ROOT, load_json, provenance, read_jsonl, save_json, sha256
from train_reviewed_fitting import finite_tree

STAGES = ['A0-fitting', 'A0-eval', 'A2-train', 'A2-fitting', 'A2-eval',
          'A3-train', 'A3-fitting', 'A3-eval']


def exact_files(directory, expected):
    directory = Path(directory)
    require(directory.is_dir() and all(p.is_file() for p in directory.iterdir())
            and {p.name for p in directory.iterdir()} == expected,
            'Incomplete or unexpected fitting artifact set.')


def rgb_pixels(path, size):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (size, size), 'Wrong fitting RGB mode or dimensions.')
        pixels = np.asarray(image)
    require(pixels.dtype == np.uint8 and pixels.shape == (size, size, 3), 'Invalid fitting RGB pixels.')
    return pixels


def checkpoint_payload(state, shapes, step, config):
    require(all(k in state for k in ('adapter', 'optimizer', 'scaler', 'rng', 'trainable_parameters'))
            and state['trainable_parameters'] == 4448 and finite_tree(state),
            'Incomplete or nonfinite checkpoint payload.')
    adapter = state['adapter']
    require(set(adapter) == set(shapes), 'Wrong adapter parameter keys.')
    for key, shape in shapes.items():
        value = adapter[key]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == torch.float32,
                'Wrong adapter parameter shape or dtype.')
    require(sum(v.numel() for v in adapter.values()) == 4448, 'Wrong adapter parameter count.')
    optimizer = state['optimizer']
    require(set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1,
            'Unexpected optimizer parameter groups.')
    group = optimizer['param_groups'][0]
    params = group['params']
    require(len(params) == len(shapes) and len(set(params)) == len(params)
            and set(optimizer['state']) == set(params), 'Optimizer does not cover every adapter parameter.')
    require(group['lr'] == config['learning_rate'] and group['weight_decay'] == config['weight_decay']
            and group.get('amsgrad') is False, 'Optimizer configuration differs from the fixed protocol.')
    for parameter, shape in zip(params, shapes.values()):
        values = optimizer['state'][parameter]
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Incomplete optimizer moment payload.')
        require(isinstance(values['step'], torch.Tensor) and values['step'].numel() == 1
                and float(values['step']) == step, 'Optimizer update count differs.')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = values[key]
            require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == torch.float32,
                    'Optimizer moment shape or dtype differs.')
        require(bool((values['exp_avg_sq'] >= 0).all()), 'Negative optimizer second moment.')
    require(config['amp'] is False and state['scaler'] == {}, 'Unexpected fixed float32 scaler state.')
    rng = state['rng']
    require(isinstance(rng, dict) and set(rng) == {'python', 'numpy', 'torch', 'cuda'}, 'Incomplete RNG payload.')
    try:
        random.Random().setstate(rng['python'])
        numpy_rng = rng['numpy']
        np.random.RandomState().set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        require(isinstance(rng['torch'], torch.Tensor) and rng['torch'].dtype == torch.uint8,
                'Invalid CPU RNG payload.')
        torch.Generator(device='cpu').set_state(rng['torch'])
        require(isinstance(rng['cuda'], list) and bool(rng['cuda'])
                and all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() > 0
                        for v in rng['cuda']), 'Missing recorded CUDA RNG bytes.')
    except (TypeError, ValueError, RuntimeError, IndexError) as error:
        raise ValueError('Invalid checkpoint RNG payload.') from error


def check_live_exit():
    """Inspect process state without invoking CUDA or nvidia-smi."""
    require(not LOCK.exists(), 'Fitting GPU lock is still present; inspect its owner.')
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid() or not (process.info['name'] or '').lower().startswith('python'):
            continue
        command = ' '.join(process.info['cmdline'] or [])
        worker = any(s in command for s in ('run_reviewed_fitting.py', 'train_reviewed_fitting.py'))
        worker = worker or ('satground.cli' in command and any(s in command for s in ('generate', 'evaluate')))
        if not worker:
            continue
        try:
            require(Path(process.cwd()).resolve() != ROOT.resolve(), 'A project fitting runner or model child remains live.')
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError('Cannot verify a possible live fitting worker.') from error


def audit(run, output, check_live=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(output.is_relative_to(run) and not output.exists(), 'Use a fresh aggregate audit output inside the private run.')
    status = load_json(run / 'status.json')
    require(status.get('status') == 'complete' and status.get('completed') == STAGES
            and status.get('current_stage') is None and status.get('child_pid') is None,
            'The serial fitting experiment is not complete.')
    identity = status['identity']
    check_pins(identity['files'])
    require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source differs from the run.')
    bundle = Path(identity['bundle'])
    require(sha256(bundle / 'bundle.json') == identity['bundle_sha256'], 'Fitting bundle identity changed.')
    record = verify_bundle(bundle)
    rows = read_jsonl(bundle / 'manifest.jsonl')
    manual = load_json(bundle / 'manual-index.json')
    ids = {r['sample_id'] for r in rows}
    require(len(rows) == len(ids) == record['fitting_count'] == 8 and set(manual) == ids,
            'Expected exactly eight reviewed fitting views.')
    protocol = record['protocol']
    require(protocol['steps'] == 200 and protocol['save_every'] == 50 and protocol['amp'] is False,
            'Unexpected fitting budget.')
    shape = (protocol['size'], protocol['size'])
    image_names, probability_names = {sid + '.png' for sid in ids}, {sid + '.npy' for sid in ids}
    for name in ('A0', 'A2', 'A3'):
        images, evaluation = run / f'{name}-fitting', run / f'{name}-eval'
        exact_files(images, image_names | {'generation.json', 'predictions.jsonl'})
        for subdir in ('masks', 'targets'):
            exact_files(evaluation / subdir, image_names)
        exact_files(evaluation / 'probabilities', probability_names)
        for sid in ids:
            rgb_pixels(images / (sid + '.png'), protocol['size'])
            target_path = evaluation / 'targets' / (sid + '.png')
            target = rgb_pixels(target_path, protocol['size'])
            original = rgb_pixels(manual[sid]['target_image'], protocol['size'])
            require(sha256(target_path) == manual[sid]['target_image_sha256']
                    and np.array_equal(target, original), 'Evaluated target bytes or pixels differ from the reviewed target.')
            binary_mask(evaluation / 'masks' / (sid + '.png'), shape)
            values = np.load(evaluation / 'probabilities' / (sid + '.npy'), allow_pickle=False)
            require(values.shape == shape and values.dtype == np.float32 and np.isfinite(values).all()
                    and ((values >= 0) & (values <= 1)).all(), 'Invalid evaluator probability array.')
    shapes = {key: tuple(value.shape) for key, value in StructuralAdapter(64, 16).state_dict().items()}
    logs, training = {}, {}
    for name in ('A2', 'A3'):
        logs[name], training[name] = training_audit(run, name, identity['training_identity'][name], protocol, ids)
        required_log_fields = {'rgb', 'lpips', 'region', 'boundary', 'total', 'gradient_norm',
                               'elapsed_seconds', 'peak_vram_mib', 'sampled_view_ids', 'step'}
        require(all(required_log_fields <= set(row) for row in logs[name]), 'Incomplete fitting loss log fields.')
        config = load_json(run / f'{name}-train/run.json')['identity']['config']
        for step in (50, 100, 150, 200):
            state = torch.load(run / f'{name}-train/step-{step:06d}.pt', map_location='cpu', weights_only=True)
            checkpoint_payload(state, shapes, step, config)
        state = torch.load(run / f'{name}-train/last.pt', map_location='cpu', weights_only=True)
        checkpoint_payload(state, shapes, 200, config)
    require([r['sampled_view_ids'] for r in logs['A2']] == [r['sampled_view_ids'] for r in logs['A3']],
            'A2/A3 sampled update schedules differ.')
    if check_live:
        check_live_exit()
    # No individual sample paths, IDs, positions or human responses enter this result.
    result = dict(status='verified', scope='training_fitting_completion_audit_only', fitting_views=8,
                  rgb_images_verified=24, evaluated_masks_verified=24, reviewed_target_copies_verified=24,
                  probability_arrays_verified=24, periodic_checkpoints_verified=8, last_checkpoints_verified=2,
                  trainable_parameters=4448, optimizer_and_rng_payloads_verified=True,
                  cuda_rng_payload_check='dtype_dimensions_and_nonempty_bytes_only',
                  cuda_rng_restore_exercised=False, gpu_resume_exercised=False,
                  matched_sample_schedule=True, source_and_input_pins_verified=True,
                  training=training, live_exit_verified=check_live, cuda_operations=0,
                  optimizer_updates_performed=0, auditor_sha256=sha256(__file__),
                  server_gate_eligible=False, held_out_generalization_evidence=False)
    save_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/reviewed-fitting200-seed17')
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-live', action='store_true')
    args = parser.parse_args()
    print(audit(args.run, args.output, args.check_live))
