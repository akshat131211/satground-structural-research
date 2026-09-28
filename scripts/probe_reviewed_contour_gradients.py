"""Fixed 24-check reviewed-contour diagnosis; never instantiate an optimizer."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import psutil
import torch
import torch.nn.functional as F

from diagnose_objective_gradients import verify_gradient_sum
from prepare_reviewed_fitting import check_pins, require, verify_bundle
from report_reviewed_fitting import training_audit
from run_reviewed_fitting import LOCK as FITTING_LOCK, other_gpu_work
from satground.backend import load_style
from satground.camera import crop_panorama
from satground.common import (MODEL_REV, ROOT, load_json, object_hash, provenance, read_jsonl,
                             safe_data_path, save_json, seed_everything, sha256, utc_now)
from satground.losses import Objective, soft_boundary, structural_losses
from satground.runs import backend_from, read_config
from satground.semantics import rgb_tensor
from train_reviewed_fitting import finite_tree

PROTOCOL = ROOT / 'configs/contour-objective-diagnosis/protocol.json'
DOCUMENTATION = ROOT / 'docs/CONTOUR_OBJECTIVE_DIAGNOSIS.md'
LOCK = ROOT / 'runs/reviewed-contour-gradient-gpu.lock'
COMPONENTS = ('rgb', 'lpips', 'region', 'boundary_positive', 'boundary_zero')


def boundary_partition(probability, building, valid):
    """Partition the original loss using its common denominator, without balancing."""
    require(probability.shape == building.shape == valid.shape and probability.ndim == 4,
            'Contour partition requires matching NCHW tensors.')
    require(bool(torch.isfinite(probability).all()) and bool(((building == 0) | (building == 1)).all())
            and bool(((valid == 0) | (valid == 1)).all()), 'Invalid partition values or reviewed labels.')
    probability = probability.clamp(1e-6, 1 - 1e-6)
    interior = -F.max_pool2d(-valid, 3, stride=1, padding=1)
    target_boundary = soft_boundary(building)
    positive = target_boundary > 0
    error = (soft_boundary(probability) - target_boundary).abs() * interior
    denominator = interior.sum().clamp_min(1)
    parts = dict(boundary_positive=(error * positive).sum() / denominator,
                 boundary_zero=(error * ~positive).sum() / denominator)
    counts = dict(valid_pixels=int(valid.sum()), boundary_valid_pixels=int(interior.sum()),
                  positive_target_contour_pixels=int((interior * positive).sum()),
                  zero_target_contour_pixels=int((interior * ~positive).sum()),
                  scored_building_pixels=int((building * valid).sum()))
    return parts, counts


def component_losses(objective, prediction, target, labels):
    probability = objective.segmenter.probabilities(prediction)[:, 2:3]
    region, boundary = structural_losses(probability, labels['building'], labels['valid'])
    partition, counts = boundary_partition(probability, labels['building'], labels['valid'])
    require(torch.allclose(partition['boundary_positive'] + partition['boundary_zero'], boundary,
                           rtol=1e-6, atol=1e-7), 'Boundary partition changes the original scalar loss.')
    return dict(rgb=F.l1_loss(prediction, target),
                lpips=objective.perceptual(prediction * 2 - 1, target * 2 - 1).mean(),
                region=region, **partition), counts


def component_weights(weights):
    return {k: weights['boundary'] if k.startswith('boundary_') else weights[k] for k in COMPONENTS}


def verify_loss_values(components, values, weights):
    require(set(components) == set(COMPONENTS) and finite_tree(components) and finite_tree(values),
            'Missing or nonfinite objective components.')
    expected = {k: components[k] for k in ('rgb', 'lpips', 'region')}
    expected['boundary'] = components['boundary_positive'] + components['boundary_zero']
    require(all(math.isclose(expected[k], values[k], rel_tol=1e-6, abs_tol=1e-7) for k in expected),
            'Component values differ from the unchanged Objective.')
    total = sum(component_weights(weights)[k] * components[k] for k in COMPONENTS)
    require(math.isclose(total, values['total'], rel_tol=1e-6, abs_tol=1e-7),
            'Weighted components differ from the unchanged Objective.')


def gradient_summary(vectors):
    require(set(vectors) == set(COMPONENTS), 'Missing gradient component.')
    values = {k: v.detach().reshape(-1).double().cpu() for k, v in vectors.items()}
    require(len({v.numel() for v in values.values()}) == 1 and finite_tree(values), 'Invalid gradient vectors.')
    values['boundary'] = values['boundary_positive'] + values['boundary_zero']
    values['A1'] = values['rgb'] + values['lpips']
    values['A2'] = values['A1'] + values['region']
    values['A3'] = values['A2'] + values['boundary']
    norms = {k: float(v.norm()) for k, v in values.items()}
    pairs = [('boundary_positive', 'boundary_zero'), ('boundary_positive', 'A2'), ('boundary_zero', 'A2'),
             ('boundary', 'rgb'), ('boundary', 'lpips'), ('boundary', 'region'), ('boundary', 'A2'), ('A2', 'A3')]
    cosines = {a + '_vs_' + b: float(torch.dot(values[a], values[b]) / (norms[a] * norms[b]))
               if norms[a] and norms[b] else None for a, b in pairs}
    ratios = dict(boundary_vs_A2=norms['boundary'] / norms['A2'] if norms['A2'] else None,
                  zero_vs_positive=norms['boundary_zero'] / norms['boundary_positive']
                  if norms['boundary_positive'] else None,
                  positive_vs_boundary=norms['boundary_positive'] / norms['boundary'] if norms['boundary'] else None)
    return dict(weighted_norms=norms, cosines=cosines, norm_ratios=ratios)


def measure(objective, prediction, target, labels, parameters, protocol):
    """Every loss stays connected to the same render graph; only measured vectors are detached."""
    losses, support = component_losses(objective, prediction, target, labels)
    weights = component_weights(protocol['weights'])
    vectors = {'image': {}, 'adapter': {}}
    for name in COMPONENTS:
        gradients = torch.autograd.grad(weights[name] * losses[name], (prediction, *parameters),
                                        retain_graph=True, allow_unused=False)
        vectors['image'][name] = gradients[0].detach().reshape(-1).cpu()
        vectors['adapter'][name] = torch.cat([g.detach().reshape(-1) for g in gradients[1:]]).cpu()
    actual_loss, values = objective(prediction, target, labels)
    raw = {k: float(v.detach()) for k, v in losses.items()}
    verify_loss_values(raw, values, protocol['weights'])
    actual = torch.autograd.grad(actual_loss, (prediction, *parameters), allow_unused=False)
    vectors['actual_image'] = actual[0].detach().reshape(-1).cpu()
    vectors['actual_adapter'] = torch.cat([g.detach().reshape(-1) for g in actual[1:]]).cpu()
    errors = {space: verify_gradient_sum(vectors['actual_' + space], sum(vectors[space].values()),
                  tolerance=protocol[space + '_gradient_relative_tolerance']) for space in ('image', 'adapter')}
    record = dict(raw_components=raw, objective_values=values, support=support,
                  gradient_sum_relative_error=errors,
                  **{space + '_gradients': gradient_summary(vectors[space]) for space in ('image', 'adapter')})
    require(finite_tree(record), 'Nonfinite probe result.')
    return record, vectors


def protocol_identity(path):
    path = Path(path).resolve()
    protocol = load_json(path)
    require(protocol['states'] == ['A0', 'A2', 'A3'] and protocol['views'] == 8 and protocol['expected_checks'] == 24
            and protocol['optimizer_updates'] == 0 and protocol['initialization_seed'] == 17
            and protocol['view_seed_base'] == 101 and protocol['parameter_changes_allowed'] is False
            and protocol['target_label_changes_allowed'] is False
            and protocol['precision'] == 'float32_with_tf32_disabled_for_diagnostic_only'
            and protocol['weights'] == {'rgb': 1., 'lpips': .1, 'region': .5, 'boundary': .5}
            and protocol['image_gradient_relative_tolerance'] == 5e-5
            and protocol['adapter_gradient_relative_tolerance'] == 2e-4, 'Different diagnostic protocol.')
    bundle, fitting = ((ROOT / protocol[k]).resolve() for k in ('bundle', 'completed_fitting_run'))
    require(bundle == (ROOT / 'data/review/reviewed-fitting8-v1').resolve()
            and fitting == (ROOT / 'runs/reviewed-fitting200-seed17').resolve(), 'Wrong reviewed diagnostic cohort.')
    record = verify_bundle(bundle)
    state = load_json(fitting / 'status.json')
    require(state['status'] == 'complete' and state['child_pid'] is None, 'Original fitting run is incomplete.')
    check_pins(state['identity']['files'])
    rows = read_jsonl(bundle / 'manifest.jsonl')
    require(len(rows) == record['fitting_count'] == 8, 'Different reviewed cohort.')
    cfg = read_config(bundle / 'A3.yaml')
    require(cfg['amp'] is False and cfg['size'] == 256 and cfg['adapter_width'] == 16, 'Unexpected renderer settings.')
    ids = {r['sample_id'] for r in rows}
    checkpoints = {n: fitting / f'{n}-train/last.pt' for n in ('A2', 'A3')}
    for name in checkpoints:
        training_audit(fitting, name, state['identity']['training_identity'][name], record['protocol'], ids)
    files = {**record['source_file_sha256'], **record['artifact_sha256'], **state['identity']['files']}
    for file in (path, DOCUMENTATION, Path(__file__).resolve(), ROOT / 'scripts/diagnose_objective_gradients.py',
                 ROOT / 'scripts/verify_training_review.py',
                 fitting / 'status.json', bundle / 'bundle.json', *checkpoints.values()):
        files[str(file)] = sha256(file)
    feature_cache = Path(cfg['feature_cache']).resolve()
    for row in rows:
        satellite = safe_data_path('data/vigor', row['satellite'])
        key = object_hash(dict(image_sha256=sha256(satellite), model=MODEL_REV, amp=False,
                               preprocessing='RGB-bicubic256-ImageNet', storage='float32'))
        cached = feature_cache / (key + '.pt')
        require(cached.is_file(), 'Prepare missing frozen scene features before freezing diagnostic inputs.')
        tensor = torch.load(cached, map_location='cpu', weights_only=True)
        require(tuple(tensor.shape) == (1, 64, 320, 320) and tensor.dtype == torch.float32 and finite_tree(tensor),
                'Invalid frozen scene feature cache.')
        files[str(cached)] = sha256(cached)
    plan = [dict(state=name, fitting_view=i + 1, sample_id=row['sample_id'], seed=protocol['view_seed_base'] + i)
            for name in protocol['states'] for i, row in enumerate(rows)]
    p = provenance()
    require(p['pipeline_source_hash'] == state['identity']['pipeline_source_hash'], 'Historical core source changed.')
    identity = dict(protocol=protocol, files=files, plan=plan, pipeline_source_hash=p['pipeline_source_hash'],
                    environment={**{k: p[k] for k in ('python', 'packages', 'code_revision', 'model_revision', 'platform')},
                                 'torch_cpu_threads': torch.get_num_threads(), 'cuda_build': torch.version.cuda})
    return protocol, bundle, cfg, rows, checkpoints, identity


def commit_record(output, ordinal, record, vectors, identity_hash):
    directory = Path(output) / 'checks' / f'{ordinal:04d}'
    require(not directory.exists(), 'Partial check outputs exist; preserve and inspect them before recovery.')
    directory.mkdir(parents=True)
    torch.save(vectors, directory / 'gradients.pt')
    save_json(directory / 'record.json', record)
    receipt = dict(ordinal=ordinal, identity_hash=identity_hash,
                   record_sha256=sha256(directory / 'record.json'), vectors_sha256=sha256(directory / 'gradients.pt'))
    save_json(directory / 'receipt.json', receipt)
    return sha256(directory / 'receipt.json')


def verified_records(output, state):
    output = Path(output)
    require(state['identity_hash'] == object_hash(state['identity']) and 0 <= state['completed'] <= 24
            and len(state['receipts']) == state['completed'], 'Invalid completed-record ledger.')
    expected = {f'{i:04d}' for i in range(1, state['completed'] + 1)}
    require(not (output / 'checks').exists() or {p.name for p in (output / 'checks').iterdir()} == expected,
            'Uncommitted partial check outputs exist; preserve them before compatible recovery.')
    protocol, plan = state['identity']['protocol'], state['identity']['plan']
    result = []
    for i, receipt_sha in enumerate(state['receipts'], 1):
        directory = output / 'checks' / f'{i:04d}'
        require(sha256(directory / 'receipt.json') == receipt_sha, 'Immutable check receipt changed.')
        receipt = load_json(directory / 'receipt.json')
        require(receipt['ordinal'] == i and receipt['identity_hash'] == state['identity_hash']
                and sha256(directory / 'record.json') == receipt['record_sha256']
                and sha256(directory / 'gradients.pt') == receipt['vectors_sha256'], 'Exact per-check hashes differ.')
        record = load_json(directory / 'record.json')
        require(record['plan'] == plan[i - 1] and record['parameters_unchanged'] and record['optimizer_updates'] == 0,
                'Check is outside the immutable zero-update schedule.')
        vectors = torch.load(directory / 'gradients.pt', map_location='cpu', weights_only=True)
        require(set(vectors) == {'image', 'adapter', 'actual_image', 'actual_adapter'}, 'Missing raw gradient payload.')
        verify_loss_values(record['raw_components'], record['objective_values'], protocol['weights'])
        for space, length in (('image', 3 * 256 * 256), ('adapter', 4448)):
            require(all(v.dtype == torch.float32 and v.shape == (length,) for v in vectors[space].values())
                    and vectors['actual_' + space].dtype == torch.float32
                    and vectors['actual_' + space].shape == (length,), 'Wrong gradient payload dimensions.')
            require(record[space + '_gradients'] == gradient_summary(vectors[space]), 'Recorded gradient statistics differ.')
            error = verify_gradient_sum(vectors['actual_' + space], sum(vectors[space].values()),
                                        tolerance=protocol[space + '_gradient_relative_tolerance'])
            require(error == record['gradient_sum_relative_error'][space], 'Recorded gradient sum error differs.')
        result.append(record)
    return result


def aggregate(records, state):
    require(len(records) == state['completed'] == state['expected'] == 24, 'Only the exact completed24 cohort is reportable.')
    def distribution(values):
        available = [x for x in values if x is not None]
        require(finite_tree(available), 'Nonfinite aggregate statistic.')
        return dict(available=len(available), unavailable=len(values) - len(available),
                    mean=float(np.mean(available)) if available else None,
                    median=float(np.median(available)) if available else None,
                    minimum=min(available) if available else None, maximum=max(available) if available else None)
    summaries = {}
    for name in ('A0', 'A2', 'A3'):
        rows = [r for r in records if r['plan']['state'] == name]
        require(len(rows) == 8, 'Incomplete model-state cohort.')
        summaries[name] = dict(views=8, support={k: sum(r['support'][k] for r in rows) for k in rows[0]['support']},
            gradients={space: {kind: {key: distribution([r[space + '_gradients'][kind][key] for r in rows])
                        for key in rows[0][space + '_gradients'][kind]} for kind in ('weighted_norms', 'cosines', 'norm_ratios')}
                       for space in ('image', 'adapter')},
            zero_boundary_gradients=sum(r['adapter_gradients']['weighted_norms']['boundary'] == 0 for r in rows),
            maximum_sum_error={space: max(r['gradient_sum_relative_error'][space] for r in rows) for space in ('image', 'adapter')})
    return dict(scope='zero_update_reviewed_training_contour_diagnosis_only', total_checks=24, views_per_state=8,
                optimizer_updates=0, parameters_unchanged=True, source_and_inputs_unchanged=True,
                weights=state['identity']['protocol']['weights'], states=summaries,
                fp32_tf32_off_diagnostic_only=True, startup_check_included_in_24=True,
                server_gate_eligible=False, held_out_generalization_evidence=False)


def diagnose(protocol_path, output, resume=False, startup_one_check=False):
    os.chdir(ROOT)
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / 'runs') and not (resume and startup_one_check), 'Keep private probe outputs in runs/.')
    protocol, bundle, cfg, rows, checkpoints, identity = protocol_identity(protocol_path)
    identity_hash = object_hash(identity)
    require(resume or not output.exists(), 'Use fresh output or compatible zero-update resume.')
    state = load_json(output / 'status.json') if resume else dict(identity=identity, identity_hash=identity_hash,
        created_utc=utc_now(), status='initializing', completed=0, expected=24, receipts=[], optimizer_updates=0,
        elapsed_seconds=0., peak_vram_mib=0., parameters_unchanged=False)
    require(state['identity'] == identity and state['identity_hash'] == identity_hash, 'Source, inputs or environment changed.')
    completed = verified_records(output, state) if resume else []
    if state['status'] == 'complete':
        require(sha256(output / 'summary.json') == state['summary_sha256']
                and load_json(output / 'summary.json') == aggregate(completed, state), 'Completed aggregate changed.')
        return state
    require(not LOCK.exists() and not FITTING_LOCK.exists() and not other_gpu_work(),
            'A project lock or GPU worker exists; inspect it before launching or resuming.')
    output.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(exist_ok=True)
    token = uuid4().hex
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(), token=token), stream)
    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    state['original_tf32_settings'] = dict(matmul=old_tf32[0], cudnn=old_tf32[1])
    begin, prior_elapsed = time.perf_counter(), state['elapsed_seconds']
    save_json(output / 'status.json', state)
    try:
        seed_everything(protocol['initialization_seed'])
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        backend = backend_from(cfg)
        initial = {k: v.detach().clone() for k, v in backend.adapter.state_dict().items()}
        backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
        weights = protocol['weights']
        objective = Objective(weights['region'], weights['boundary'], weights['lpips'], cfg['training_segmenter_revision'], 'legacy')
        parameters = tuple(backend.adapter.parameters())
        require(sum(p.numel() for p in parameters) == 4448, 'Different adapter size.')
        torch.cuda.reset_peak_memory_stats()
        for ordinal, plan in enumerate(identity['plan'], 1):
            if ordinal <= state['completed']:
                continue
            check_pins(identity['files'])
            state.update(status='running', current_check=ordinal, updated_utc=utc_now())
            save_json(output / 'status.json', state)
            if plan['state'] == 'A0':
                backend.adapter.load_state_dict(initial)
            else:
                checkpoint = torch.load(checkpoints[plan['state']], map_location='cpu', weights_only=True)
                require(checkpoint['step'] == 200, 'Wrong final fitting checkpoint.')
                backend.adapter.load_state_dict(checkpoint['adapter'])
            before = {k: v.detach().clone() for k, v in backend.adapter.state_dict().items()}
            row = rows[plan['fitting_view'] - 1]
            raw = backend.cached_features(safe_data_path('data/vigor', row['satellite']), cfg['feature_cache'])
            if plan['state'] == 'A0':
                with torch.no_grad():
                    require(torch.equal(backend.adapter(raw), raw), 'A0 adapter does not preserve baseline features.')
            with Image.open(safe_data_path('data/vigor', row['panorama'])) as image:
                target = rgb_tensor(crop_panorama(image.convert('RGB'), row['yaw'], row['pitch'], row['fov'], cfg['size']), 'cuda')
            with np.load(bundle / 'labels' / (row['sample_id'] + '.npz'), allow_pickle=False) as values:
                labels = {k: torch.from_numpy(values[k]).cuda()[None, None] for k in ('building', 'valid')}
            seed_everything(plan['seed'])
            prediction = backend.render(raw, row, adapted=True)
            record, vectors = measure(objective, prediction, target, labels, parameters, protocol)
            backend.assert_frozen()
            require(all(torch.equal(v, before[k]) for k, v in backend.adapter.state_dict().items())
                    and all(p.grad is None for p in parameters), 'Probe changed adapter parameters or stored gradients.')
            require(all(not p.requires_grad and p.grad is None for module in (objective.perceptual, objective.segmenter.model)
                        for p in module.parameters()), 'Reference loss modules are not frozen.')
            check_pins(identity['files'])
            require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source changed during probe.')
            record.update(plan=plan, parameters_unchanged=True, optimizer_updates=0, frozen_models_verified=True)
            state['receipts'].append(commit_record(output, ordinal, record, vectors, identity_hash))
            state.update(completed=ordinal, elapsed_seconds=prior_elapsed + time.perf_counter() - begin,
                         peak_vram_mib=max(state['peak_vram_mib'], torch.cuda.max_memory_allocated() / 2**20),
                         parameters_unchanged=True, updated_utc=utc_now())
            save_json(output / 'status.json', state)
            print(f"Check {ordinal}/24, {plan['state']} fitting view {plan['fitting_view']}, finite exact decomposition, zero updates", flush=True)
            del prediction, record, vectors, raw, target, labels, before
            if startup_one_check:
                state.update(status='startup_complete', current_check=None)
                save_json(output / 'status.json', state)
                return state
        records = verified_records(output, state)
        check_pins(identity['files'])
        save_json(output / 'summary.json', aggregate(records, state))
        state.update(status='complete', current_check=None, summary_sha256=sha256(output / 'summary.json'), updated_utc=utc_now())
        save_json(output / 'status.json', state)
        return state
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', updated_utc=utc_now())
        save_json(output / 'status.json', state)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        if LOCK.exists() and load_json(LOCK).get('token') == token:
            LOCK.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(PROTOCOL))
    parser.add_argument('--output', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--startup-one-check', action='store_true')
    args = parser.parse_args()
    result = diagnose(args.protocol, args.output, args.resume, args.startup_one_check)
    print({k: result[k] for k in ('status', 'completed', 'expected', 'optimizer_updates')})
