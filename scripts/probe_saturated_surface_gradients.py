"""Eight separate saturated-v2 A0 feasibility checks; never instantiate an optimizer."""
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
from prepare_reviewed_fitting import check_pins, require
import probe_surface_candidate_gradients as previous
from probe_reviewed_contour_gradients import LOCK as HISTORICAL_LOCK
from run_reviewed_fitting import LOCK as FITTING_LOCK, other_gpu_work
from saturated_surface_objective import (COEFFICIENT, MASS_FLOOR, POLICY, RADIUS,
                                        prepare_surface_labels, valid_surface_loss)
from satground.backend import load_style
from satground.camera import crop_panorama
from satground.common import (ROOT, load_json, object_hash, provenance, safe_data_path,
                             save_json, seed_everything, sha256, utc_now)
from satground.losses import Objective, structural_losses
from satground.runs import backend_from
from satground.semantics import rgb_tensor
from train_reviewed_fitting import finite_tree


PROTOCOL = ROOT / 'configs/contour-objective-diagnosis/saturated-protocol.json'
OUTPUT = ROOT / 'runs/reviewed-saturated-surface8'
LOCK = ROOT / 'runs/reviewed-saturated-surface-gpu.lock'
COMPONENTS = previous.COMPONENTS
SUPPORT_KEYS = ('accepted_valid_pixels', 'domain_pixels', 'historical_positive_contour_pixels',
    'interface_source_pixels', 'historical_contour_pixels_retained_as_sources',
    'historical_contour_pixels_not_sources', 'domain_components', 'domain_components_without_observed_interface',
    'near_band_pixels', 'surface_scored_pixels', 'saturated_distance_pixels', 'beyond_radius_saturated_pixels')


def validate_support(support):
    required_arrays = ('class_scored_pixels', 'near_band_class_pixels', 'class_weight_mass', 'class_denominators')
    require(set(SUPPORT_KEYS).issubset(support) and 'band_pixels' not in support and
        all(key in support and len(support[key]) == 2 for key in required_arrays) and finite_tree(support),
        'Saturated support must retain distinct near-band, total and class counts.')
    require(all(type(support[k]) is int and support[k] >= 0 for k in SUPPORT_KEYS) and
        0 <= support['near_band_pixels'] <= support['surface_scored_pixels'] <= support['domain_pixels'] <= support['accepted_valid_pixels'] and
        support['beyond_radius_saturated_pixels'] == support['surface_scored_pixels'] - support['near_band_pixels'] and
        support['beyond_radius_saturated_pixels'] <= support['saturated_distance_pixels'] <= support['surface_scored_pixels'] and
        sum(support['class_scored_pixels']) == support['surface_scored_pixels'] and
        sum(support['near_band_class_pixels']) == support['near_band_pixels'] and
        all(m >= 0 and math.isclose(d, max(m, MASS_FLOOR), rel_tol=1e-6, abs_tol=1e-7)
            for m, d in zip(support['class_weight_mass'], support['class_denominators'])),
        'Inconsistent saturated support or class normalization.')


def validate_protocol(protocol):
    expected = dict(study='reviewed_saturated_surface_feasibility',
        historical_protocol='configs/contour-objective-diagnosis/protocol.json',
        completed_historical_probe='runs/reviewed-contour-objective-diagnosis24',
        completed_failed_candidate_probe='runs/reviewed-surface-candidate8',
        controlled_report='reports/saturated-surface-diagnosis/controlled-cases.json',
        documentation='docs/VALID_SURFACE_SATURATION_REVISION.md',
        bundle='data/review/reviewed-fitting8-v1', states=['A0'], views=8, expected_checks=8,
        optimizer_updates=0, initialization_seed=17, view_seed_base=101,
        weights=dict(rgb=1., lpips=.1, region=.5, surface=COEFFICIENT), surface_policy=POLICY,
        radius=RADIUS, class_mass_floor=MASS_FLOOR, required_supported_nonzero_finite_adapter_views=6,
        precision='float32_with_tf32_disabled_for_diagnostic_only',
        image_gradient_relative_tolerance=5e-5, adapter_gradient_relative_tolerance=2e-4,
        parameter_changes_allowed=False, target_label_changes_allowed=False,
        server_gate_eligible=False, held_out_generalization_evidence=False)
    require(protocol == expected, 'Different fixed saturated feasibility protocol.')


def validate_previous(output, identity):
    """Preserve and reverify all eight v1 receipts and their historical 24 dependency."""
    output = Path(output)
    state = load_json(output / 'status.json')
    require(state['status'] == 'complete' and state['completed'] == state['expected'] == 8
        and state['identity'] == identity and state['identity_hash'] == object_hash(identity),
        'Prior v1 candidate probe is incomplete or its identity differs.')
    records = previous.verified_records(output, state)
    require(sha256(output / 'summary.json') == state['summary_sha256'] and
        load_json(output / 'summary.json') == previous.aggregate(records, state), 'Prior v1 aggregate differs.')
    return state, records


def validate_controlled(path, reproduce=None):
    """Pure CPU prerequisite before a lock, renderer or GPU allocation is created."""
    if reproduce is None:
        from compare_saturated_surface_cases import build_report
        reproduce = build_report
    report = load_json(path)
    require(finite_tree(report) and report.get('training_clearance') is True,
            'Saturated controlled cases do not grant prerequisite clearance.')
    require(report == reproduce(), 'Saturated controlled report does not reproduce exactly.')
    return report


def protocol_identity(path):
    path = Path(path).resolve()
    protocol = load_json(path)
    validate_protocol(protocol)
    old_protocol, bundle, cfg, rows, history, historical, old_identity = previous.protocol_identity(previous.PROTOCOL)
    require(all(protocol[k] == old_protocol[k] for k in ('bundle', 'initialization_seed', 'view_seed_base',
        'precision', 'weights', 'image_gradient_relative_tolerance', 'adapter_gradient_relative_tolerance')),
        'Saturated and prior render settings differ.')
    old_run = ROOT / protocol['completed_failed_candidate_probe']
    old_state, _ = validate_previous(old_run, old_identity)
    controlled_path = ROOT / protocol['controlled_report']
    controlled = validate_controlled(controlled_path)
    files = dict(old_identity['files'])
    for file in (path, ROOT / protocol['documentation'], Path(__file__).resolve(),
        ROOT / 'scripts/saturated_surface_objective.py', ROOT / 'scripts/compare_saturated_surface_cases.py',
        ROOT / 'scripts/probe_surface_candidate_gradients.py', ROOT / 'scripts/contour_objective_cases.py',
        ROOT / 'scripts/compare_surface_controlled_cases.py',
        controlled_path, old_run / 'status.json', old_run / 'summary.json'):
        require(file.is_file(), 'Missing frozen saturated prerequisite source or evidence.')
        files[str(file.resolve())] = sha256(file)
    for relative, expected_hash in controlled['source_hashes'].items():
        file = (ROOT / relative).resolve()
        require(file.is_relative_to(ROOT.resolve()) and file.is_file() and sha256(file) == expected_hash,
                'A reproduced controlled-case source hash differs or escapes the workspace.')
        files[str(file)] = expected_hash
    # Preserve the concrete original and failed-v1 controlled evidence too.
    for file in (ROOT / 'configs/contour-objective-diagnosis/protocol.json',
                 ROOT / 'reports/contour-objective-diagnosis/controlled-cases.json',
                 ROOT / 'reports/contour-objective-diagnosis/surface-controlled-cases.json'):
        require(file.is_file(), 'Missing historical controlled-case input.')
        files[str(file.resolve())] = sha256(file)
    for directory in sorted((old_run / 'checks').iterdir()):
        for name in ('receipt.json', 'record.json', 'gradients.pt'):
            file = directory / name
            files[str(file.resolve())] = sha256(file)
    identity = dict(protocol=protocol, files=files, plan=old_identity['plan'],
        pipeline_source_hash=old_identity['pipeline_source_hash'], environment=old_identity['environment'],
        prior_candidate_identity_hash=old_state['identity_hash'], prior_candidate_summary_sha256=old_state['summary_sha256'],
        historical_identity_hash=old_identity['historical_identity_hash'], controlled_report_sha256=sha256(controlled_path),
        controlled_report_object_hash=object_hash(controlled), controlled_prerequisite_clearance=True)
    return protocol, bundle, cfg, rows, history, historical, identity


def measure(objective, prediction, target, labels, prepared, parameters, protocol, historical):
    """Keep all terms attached to the original render graph; detach measured vectors only."""
    probability = objective.segmenter.probabilities(prediction)[:, 2:3]
    region, _ = structural_losses(probability, labels['building'], labels['valid'])
    losses = dict(rgb=F.l1_loss(prediction, target),
        lpips=objective.perceptual(prediction * 2 - 1, target * 2 - 1).mean(), region=region,
        surface=valid_surface_loss(probability, prepared))
    vectors = {'image': {}, 'adapter': {}}
    weights = protocol['weights']
    for name in COMPONENTS:
        grad = torch.autograd.grad(weights[name] * losses[name], (prediction, *parameters), retain_graph=True, allow_unused=False)
        vectors['image'][name] = grad[0].detach().reshape(-1).cpu()
        vectors['adapter'][name] = torch.cat([g.detach().reshape(-1) for g in grad[1:]]).cpu()
    actual_A2, values = objective(prediction, target, labels)
    raw = {k: float(v.detach()) for k, v in losses.items()}
    previous.verify_loss_values(raw, values, weights)
    require(all(math.isclose(raw[k], historical['record']['raw_components'][k], rel_tol=1e-6, abs_tol=1e-7)
        for k in ('rgb', 'lpips', 'region')), 'Shared A0 forward losses differ from historical probe.')
    for kind, loss, retain in (('A2', actual_A2, True), ('combined', actual_A2 + weights['surface'] * losses['surface'], False)):
        grad = torch.autograd.grad(loss, (prediction, *parameters), retain_graph=retain, allow_unused=False)
        vectors[f'actual_{kind}_image'] = grad[0].detach().reshape(-1).cpu()
        vectors[f'actual_{kind}_adapter'] = torch.cat([g.detach().reshape(-1) for g in grad[1:]]).cpu()
    errors, baseline_errors, historical_errors = {}, {}, {}
    for space in ('image', 'adapter'):
        base = sum(vectors[space][k] for k in ('rgb', 'lpips', 'region'))
        tolerance = protocol[space + '_gradient_relative_tolerance']
        baseline_errors[space] = verify_gradient_sum(vectors[f'actual_A2_{space}'], base, tolerance)
        errors[space] = verify_gradient_sum(vectors[f'actual_combined_{space}'], base + vectors[space]['surface'], tolerance)
        old = sum(historical['vectors'][space][k] for k in ('rgb', 'lpips', 'region'))
        historical_errors[space] = previous.relative_error(vectors[f'actual_A2_{space}'], old)
    record = dict(raw_components=raw, A2_objective_values=values, support=prepared['counts'],
        surface_provenance=prepared['provenance'], gradient_sum_relative_error=errors,
        A2_gradient_sum_relative_error=baseline_errors, historical_A2_gradient_relative_error=historical_errors,
        **{space + '_gradients': previous.gradient_summary(vectors[space]) for space in ('image', 'adapter')})
    validate_support(record['support'])
    require(finite_tree(record), 'Nonfinite saturated feasibility result.')
    return record, vectors


def verified_records(output, state):
    output = Path(output)
    require(state['identity_hash'] == object_hash(state['identity']) and state['expected'] == 8 and
        0 <= state['completed'] <= 8 and len(state['receipts']) == state['completed'], 'Invalid saturated receipt ledger.')
    expected = {f'{i:04d}' for i in range(1, state['completed'] + 1)}
    require(not (output / 'checks').exists() or {p.name for p in (output / 'checks').iterdir()} == expected,
        'Uncommitted partial saturated checks exist; preserve them before recovery.')
    protocol, plan = state['identity']['protocol'], state['identity']['plan']
    require(len(plan) == 8 and all(p['state'] == 'A0' and p['fitting_view'] == i + 1 and p['seed'] == 101 + i
        for i, p in enumerate(plan)), 'Different saturated eight-check schedule.')
    result = []
    for i, receipt_hash in enumerate(state['receipts'], 1):
        directory = output / 'checks' / f'{i:04d}'
        require(sha256(directory / 'receipt.json') == receipt_hash, 'Saturated receipt changed.')
        receipt = load_json(directory / 'receipt.json')
        require(receipt['ordinal'] == i and receipt['identity_hash'] == state['identity_hash'] and
            sha256(directory / 'record.json') == receipt['record_sha256'] and
            sha256(directory / 'gradients.pt') == receipt['vectors_sha256'], 'Saturated check hashes differ.')
        record = load_json(directory / 'record.json')
        require(record['plan'] == plan[i - 1] and record['parameters_unchanged'] and record['frozen_models_verified']
            and record['optimizer_updates'] == 0 and finite_tree(record), 'Saturated check violates the zero-update schedule.')
        previous.verify_loss_values(record['raw_components'], record['A2_objective_values'], protocol['weights'])
        policy = record['surface_provenance']
        require(policy['policy'] == POLICY and policy['radius'] == RADIUS and policy['mass_floor'] == MASS_FLOOR
            and policy['coefficient'] == COEFFICIENT, 'Wrong saturated surface policy.')
        validate_support(record['support'])
        vectors = torch.load(directory / 'gradients.pt', map_location='cpu', weights_only=True)
        require(set(vectors) == {'image', 'adapter', 'actual_A2_image', 'actual_A2_adapter',
            'actual_combined_image', 'actual_combined_adapter'}, 'Missing saturated gradient payload.')
        for space, length in (('image', 3 * 256 * 256), ('adapter', 4448)):
            require(set(vectors[space]) == set(COMPONENTS) and
                all(v.dtype == torch.float32 and v.shape == (length,) for v in vectors[space].values()) and
                all(vectors[f'actual_{kind}_{space}'].dtype == torch.float32 and
                    vectors[f'actual_{kind}_{space}'].shape == (length,) for kind in ('A2', 'combined')),
                'Wrong saturated gradient dimensions.')
            require(record[space + '_gradients'] == previous.gradient_summary(vectors[space]), 'Saturated gradient statistics differ.')
            base = sum(vectors[space][k] for k in ('rgb', 'lpips', 'region'))
            for kind, expected_vector, key in (('A2', base, 'A2_gradient_sum_relative_error'),
                ('combined', base + vectors[space]['surface'], 'gradient_sum_relative_error')):
                error = verify_gradient_sum(vectors[f'actual_{kind}_{space}'], expected_vector,
                    protocol[space + '_gradient_relative_tolerance'])
                require(error == record[key][space], 'Saturated gradient sum error differs.')
        result.append(record)
    return result


def aggregate(records, state):
    require(len(records) == state['completed'] == state['expected'] == 8, 'Only all eight saturated checks are reportable.')
    def distribution(values):
        available = [v for v in values if v is not None]
        require(finite_tree(available), 'Nonfinite saturated aggregate.')
        return dict(available=len(available), unavailable=len(values) - len(available),
            mean=float(np.mean(available)) if available else None, median=float(np.median(available)) if available else None,
            minimum=min(available) if available else None, maximum=max(available) if available else None)
    support = {k: sum(r['support'][k] for r in records) for k in SUPPORT_KEYS}
    for key in ('class_weight_mass', 'class_scored_pixels', 'near_band_class_pixels'):
        support[key] = [sum(r['support'][key][c] for r in records) for c in (0, 1)]
    support['class_mass_floor_active_views'] = [sum(r['support']['class_weight_mass'][c] < MASS_FLOOR for r in records) for c in (0, 1)]
    supported = sum(r['support']['surface_support_present'] for r in records)
    usable = sum(r['support']['surface_support_present'] and r['adapter_gradients']['weighted_norms']['surface'] > 0
        and finite_tree(r['adapter_gradients']) for r in records)
    required = state['identity']['protocol']['required_supported_nonzero_finite_adapter_views']
    return dict(scope='zero_update_reviewed_training_saturated_surface_feasibility_only', total_checks=8, states=['A0'],
        optimizer_updates=0, parameters_unchanged=True, frozen_models_verified=True, source_and_inputs_unchanged=True,
        weights=state['identity']['protocol']['weights'], policy=POLICY, radius=RADIUS, class_mass_floor=MASS_FLOOR,
        support=support, supported_views=supported, unsupported_views=8 - supported,
        supported_nonzero_finite_adapter_views=usable, required_supported_nonzero_finite_adapter_views=required,
        feasibility_rail_passed=usable >= required, controlled_prerequisite_clearance=True,
        training_clearance=False, feasibility_rail_is_not_training_clearance=True, method_selected=False,
        label_accuracy_verified=False, held_out_generalization_evidence=False, server_gate_eligible=False,
        gradients={space: {kind: {key: distribution([r[space + '_gradients'][kind][key] for r in records])
            for key in records[0][space + '_gradients'][kind]} for kind in ('weighted_norms', 'cosines', 'norm_ratios')}
            for space in ('image', 'adapter')},
        maximum_gradient_sum_error={space: max(r['gradient_sum_relative_error'][space] for r in records) for space in ('image', 'adapter')},
        maximum_A2_gradient_sum_error={space: max(r['A2_gradient_sum_relative_error'][space] for r in records) for space in ('image', 'adapter')},
        historical_A2_gradient_relative_errors={space: distribution([r['historical_A2_gradient_relative_error'][space] for r in records])
            for space in ('image', 'adapter')}, historical_gradients_bitwise_reproducible_claim=False,
        fp32_tf32_off_diagnostic_only=True, startup_check_included_in_eight=True)


def diagnose(protocol_path, output=OUTPUT, resume=False, startup_one_check=False):
    os.chdir(ROOT)
    output = Path(output).resolve()
    require(output == OUTPUT.resolve() and not (resume and startup_one_check), 'Use the fixed private saturated eight-check path.')
    protocol, bundle, cfg, rows, history, historical, identity = protocol_identity(protocol_path)
    identity_hash = object_hash(identity)
    require(resume or not output.exists(), 'Use fresh saturated output or compatible resume.')
    state = load_json(output / 'status.json') if resume else dict(identity=identity, identity_hash=identity_hash,
        created_utc=utc_now(), status='initializing', completed=0, expected=8, receipts=[], optimizer_updates=0,
        elapsed_seconds=0., peak_vram_mib=0., parameters_unchanged=False)
    require(state['identity'] == identity and state['identity_hash'] == identity_hash, 'Saturated sources, inputs or environment changed.')
    completed = verified_records(output, state) if resume else []
    if state['status'] == 'complete':
        require(sha256(output / 'summary.json') == state['summary_sha256'] and
            load_json(output / 'summary.json') == aggregate(completed, state), 'Saturated completed aggregate changed.')
        return state
    require(not LOCK.exists() and not previous.LOCK.exists() and not HISTORICAL_LOCK.exists() and
        not FITTING_LOCK.exists() and not other_gpu_work(), 'A project lock or GPU worker exists; inspect before saturated launch or resume.')
    output.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(exist_ok=True)
    token = uuid4().hex
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(), token=token), stream)
    original_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    state['original_tf32_settings'] = dict(matmul=original_tf32[0], cudnn=original_tf32[1])
    begin, prior_elapsed = time.perf_counter(), state['elapsed_seconds']
    save_json(output / 'status.json', state)
    try:
        seed_everything(protocol['initialization_seed'])
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        backend = backend_from(cfg)
        before = {k: v.detach().clone() for k, v in backend.adapter.state_dict().items()}
        backend.set_style(load_style(cfg['style'], sha256(cfg['train_manifest'])))
        objective = Objective(protocol['weights']['region'], 0, protocol['weights']['lpips'], cfg['training_segmenter_revision'], 'legacy')
        parameters = tuple(backend.adapter.parameters())
        require(sum(p.numel() for p in parameters) == 4448, 'Different saturated adapter size.')
        torch.cuda.reset_peak_memory_stats()
        for ordinal, plan in enumerate(identity['plan'], 1):
            if ordinal <= state['completed']:
                continue
            check_pins(identity['files'])
            state.update(status='running', current_check=ordinal, updated_utc=utc_now())
            save_json(output / 'status.json', state)
            row = rows[plan['fitting_view'] - 1]
            raw = backend.cached_features(safe_data_path('data/vigor', row['satellite']), cfg['feature_cache'])
            with torch.no_grad():
                require(torch.equal(backend.adapter(raw), raw), 'Saturated A0 adapter changed baseline features.')
            with Image.open(safe_data_path('data/vigor', row['panorama'])) as image:
                target = rgb_tensor(crop_panorama(image.convert('RGB'), row['yaw'], row['pitch'], row['fov'], cfg['size']), 'cuda')
            with np.load(bundle / 'labels' / (row['sample_id'] + '.npz'), allow_pickle=False) as values:
                prepared = prepare_surface_labels(values['building'], values['valid'])
                labels = {k: torch.from_numpy(values[k]).cuda()[None, None] for k in ('building', 'valid')}
            reference = dict(record=historical[ordinal - 1], vectors=torch.load(
                history / 'checks' / f'{ordinal:04d}' / 'gradients.pt', map_location='cpu', weights_only=True))
            seed_everything(plan['seed'])
            prediction = backend.render(raw, row, adapted=True)
            record, vectors = measure(objective, prediction, target, labels, prepared, parameters, protocol, reference)
            backend.assert_frozen()
            require(all(torch.equal(v, before[k]) for k, v in backend.adapter.state_dict().items()) and
                all(p.grad is None for p in parameters), 'Saturated probe changed parameters or stored gradients.')
            require(all(not p.requires_grad and p.grad is None for module in (objective.perceptual, objective.segmenter.model)
                for p in module.parameters()), 'Saturated reference modules are not frozen.')
            check_pins(identity['files'])
            require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source changed during saturated probe.')
            record.update(plan=plan, parameters_unchanged=True, optimizer_updates=0, frozen_models_verified=True)
            state['receipts'].append(previous.commit_record(output, ordinal, record, vectors, identity_hash))
            state.update(completed=ordinal, elapsed_seconds=prior_elapsed + time.perf_counter() - begin,
                peak_vram_mib=max(state['peak_vram_mib'], torch.cuda.max_memory_allocated() / 2**20),
                parameters_unchanged=True, updated_utc=utc_now())
            save_json(output / 'status.json', state)
            print(f'Check {ordinal}/8, A0 fitting view {ordinal}, finite saturated decomposition, zero updates', flush=True)
            del prediction, record, vectors, raw, target, labels, prepared, reference
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
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_tf32
        if LOCK.exists() and load_json(LOCK).get('token') == token:
            LOCK.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(PROTOCOL))
    parser.add_argument('--output', default=str(OUTPUT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--startup-one-check', action='store_true')
    args = parser.parse_args()
    result = diagnose(args.protocol, args.output, args.resume, args.startup_one_check)
    print({k: result[k] for k in ('status', 'completed', 'expected', 'optimizer_updates')})
