"""CPU-only completion audit for the fixed historical24 and surface8 probes.

No imagery, model inference, optimizer, mask edits or new candidate are involved.
Only numerical aggregates are written, to a fresh private directory under runs/.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import contour_objective_cases as controlled
import compare_surface_controlled_cases as compared
import probe_reviewed_contour_gradients as historical
import probe_surface_candidate_gradients as candidate
from prepare_reviewed_fitting import check_pins, require
from run_reviewed_fitting import LOCK as FITTING_LOCK, other_gpu_work
from satground.common import ROOT, load_json, object_hash, save_json, sha256
from train_reviewed_fitting import finite_tree
from valid_surface_objective import POLICY, prepare_surface_labels


HISTORICAL_PROTOCOL = historical.PROTOCOL
CANDIDATE_PROTOCOL = candidate.PROTOCOL
HISTORICAL_OUTPUT = ROOT / 'runs/reviewed-contour-objective-diagnosis24'
CANDIDATE_OUTPUT = candidate.OUTPUT
CONTROLLED_REPORT = ROOT / 'reports/contour-objective-diagnosis/controlled-cases.json'
SURFACE_REPORT = ROOT / 'reports/contour-objective-diagnosis/surface-controlled-cases.json'


def require_idle():
    require(not any(path.exists() for path in (historical.LOCK, candidate.LOCK, FITTING_LOCK)),
            'A project GPU lock remains; complete and exit the probes before reporting.')
    require(not other_gpu_work(), 'Another Python/GPU worker remains; inspect its state before reporting.')


def completed_records(output, state, identity, count, module):
    require(state.get('status') == 'complete' and state.get('current_check') is None
            and state.get('completed') == state.get('expected') == count
            and state.get('optimizer_updates') == 0 and state.get('parameters_unchanged') is True,
            'Probe is incomplete or violates the zero-update completion rule.')
    require(state['identity'] == identity and state['identity_hash'] == object_hash(identity),
            'Current probe protocol/source/input/environment identity differs.')
    check_pins(identity['files'])
    records = module.verified_records(output, state)
    require(len(records) == count and finite_tree(records)
            and all(r.get('parameters_unchanged') is True and r.get('optimizer_updates') == 0
                    and r.get('frozen_models_verified') is True for r in records),
            'A finite frozen zero-update check is missing.')
    aggregate = module.aggregate(records, state)
    summary = Path(output) / 'summary.json'
    require(sha256(summary) == state['summary_sha256'] and load_json(summary) == aggregate,
            'Stored completed aggregate differs from its raw records.')
    return records, aggregate


def verify_schedules(rows, historical_records, candidate_records):
    require(len(rows) == 8 and len(historical_records) == 24 and len(candidate_records) == 8,
            'Require exactly the fixed eight views and 32 checks.')
    for state_index, name in enumerate(('A0', 'A2', 'A3')):
        for i, row in enumerate(rows):
            expected = dict(state=name, fitting_view=i + 1, sample_id=row['sample_id'], seed=101 + i)
            require(historical_records[state_index * 8 + i]['plan'] == expected,
                    'Historical state/view/seed schedule differs.')
            if name == 'A0':
                require(candidate_records[i]['plan'] == expected,
                        'Candidate and historical eight-view seed schedules differ.')


def verify_mask_support(bundle, rows, historical_records, candidate_records):
    """Read exact frozen NPZ arrays; retain all spatial data/provenance privately."""
    for i, row in enumerate(rows):
        with np.load(Path(bundle) / 'labels' / (row['sample_id'] + '.npz'), allow_pickle=False) as values:
            building, valid = values['building'].copy(), values['valid'].copy()
        require(building.shape == valid.shape == (256, 256), 'Wrong frozen reviewed mask dimensions.')
        prepared = prepare_surface_labels(building, valid)
        record = candidate_records[i]
        require(record['support'] == prepared['counts']
                and record['surface_provenance'] == prepared['provenance'],
                'Recorded candidate support or target provenance differs from the exact reviewed masks.')
        _, counts = historical.boundary_partition(torch.zeros(1, 1, 256, 256),
            torch.from_numpy(building.copy())[None, None], torch.from_numpy(valid.copy())[None, None])
        require(all(historical_records[offset + i]['support'] == counts for offset in (0, 8, 16)),
                'Historical target/support counts differ from the exact reviewed masks.')


def verify_historical_comparisons(historical_output, candidate_output, historical_records, candidate_records):
    """Recompute candidate-versus-historical A2 gradient errors from raw CPU tensors."""
    for i, record in enumerate(candidate_records, 1):
        old = torch.load(Path(historical_output) / 'checks' / f'{i:04d}' / 'gradients.pt',
                         map_location='cpu', weights_only=True)
        new = torch.load(Path(candidate_output) / 'checks' / f'{i:04d}' / 'gradients.pt',
                         map_location='cpu', weights_only=True)
        require(all(np.isclose(record['raw_components'][k], historical_records[i - 1]['raw_components'][k],
                               rtol=1e-6, atol=1e-7) for k in ('rgb', 'lpips', 'region')),
                'Shared A0 common forward losses differ from the historical probe.')
        for space in ('image', 'adapter'):
            reference = sum(old[space][k] for k in ('rgb', 'lpips', 'region'))
            error = candidate.relative_error(new[f'actual_A2_{space}'], reference)
            require(error == record['historical_A2_gradient_relative_error'][space],
                    'Recorded historical A2 gradient comparison differs from its raw tensors.')


def verify_controlled_reports():
    historical_cases, surface_cases = load_json(CONTROLLED_REPORT), load_json(SURFACE_REPORT)
    require(historical_cases == controlled.build_report() and surface_cases == compared.build_comparison(),
            'Fixed CPU controlled reports/source hashes are not reproducible.')
    require(historical_cases['synthetic_case_count'] == surface_cases['fixed_synthetic_case_count'] == 52
            and historical_cases['frozen_case_identity_sha256'] == surface_cases['fixed_case_identity_sha256']
            == compared.FROZEN_CASE_IDENTITY and finite_tree((historical_cases, surface_cases)),
            'Different or nonfinite fixed 52-case cohort.')
    require(surface_cases['prior_gate']['method'] == POLICY
            and surface_cases.get('training_clearance') is False
            and surface_cases.get('harmful_far_combined_ordering') is True
            and surface_cases.get('no_training_on_harmful_controlled_behavior') is True
            and bool(surface_cases.get('training_blockers'))
            and surface_cases['optimizer_updates'] == 0 and surface_cases['gpu_used'] is False,
            'The fixed v1 controlled-case blocker must remain explicit; no training is cleared.')
    return historical_cases, surface_cases


def interpretation(historical_aggregate, candidate_aggregate, surface_cases):
    """Counts/statistics only: no per-view mask hashes or identities are emitted."""
    states = {}
    for name, result in historical_aggregate['states'].items():
        stats = result['gradients']['adapter']
        states[name] = dict(
            median_zero_to_positive_contour_gradient_norm=stats['norm_ratios']['zero_vs_positive']['median'],
            undefined_zero_to_positive_ratios=stats['norm_ratios']['zero_vs_positive']['unavailable'],
            median_positive_vs_zero_contour_cosine=stats['cosines']['boundary_positive_vs_boundary_zero']['median'],
            median_boundary_vs_A2_cosine=stats['cosines']['boundary_vs_A2']['median'],
            zero_boundary_adapter_views=result['zero_boundary_gradients'])
    return dict(historical_adapter_gradient_statistics=states,
        candidate_supported_views=candidate_aggregate['supported_views'],
        candidate_supported_nonzero_finite_adapter_views=candidate_aggregate['supported_nonzero_finite_adapter_views'],
        candidate_feasibility_rail_passed=candidate_aggregate['feasibility_rail_passed'],
        near_band_controlled_screen_passed=surface_cases['controlled_gate_passed'],
        harmful_far_combined_ordering=True, training_clearance=False,
        training_blockers=surface_cases['training_blockers'],
        gradient_conflict_is_not_a_causal_or_efficacy_claim=True,
        historical_bitwise_gradient_reproducibility_claim=False)


def report(output):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / 'runs') and output != (ROOT / 'runs').resolve(),
            'Keep completion outputs under a fresh private runs/ directory.')
    require(not output.exists(), 'Use a fresh output; preserve previous completion evidence.')
    require_idle()
    hp, bundle, _, rows, _, historical_identity = historical.protocol_identity(HISTORICAL_PROTOCOL)
    cp, candidate_bundle, _, candidate_rows, history, _, candidate_identity = candidate.protocol_identity(CANDIDATE_PROTOCOL)
    require(Path(history).resolve() == HISTORICAL_OUTPUT.resolve()
            and Path(bundle).resolve() == Path(candidate_bundle).resolve() and rows == candidate_rows,
            'The historical/candidate frozen cohort differs.')
    require(hp['optimizer_updates'] == cp['optimizer_updates'] == 0
            and cp['historical_protocol'] == str(HISTORICAL_PROTOCOL.relative_to(ROOT)).replace('\\', '/'),
            'The candidate does not refer to the fixed historical zero-update protocol.')
    old_state, new_state = load_json(HISTORICAL_OUTPUT / 'status.json'), load_json(CANDIDATE_OUTPUT / 'status.json')
    old_records, old_aggregate = completed_records(HISTORICAL_OUTPUT, old_state, historical_identity, 24, historical)
    new_records, new_aggregate = completed_records(CANDIDATE_OUTPUT, new_state, candidate_identity, 8, candidate)
    verify_schedules(rows, old_records, new_records)
    verify_mask_support(bundle, rows, old_records, new_records)
    verify_historical_comparisons(HISTORICAL_OUTPUT, CANDIDATE_OUTPUT, old_records, new_records)
    original_cases, surface_cases = verify_controlled_reports()
    check_pins(historical_identity['files'])
    check_pins(candidate_identity['files'])
    require_idle()
    audit = dict(scope='completed_zero_update_reviewed_training_objective_diagnosis_only',
        total_finite_checks=32, historical_checks=24, candidate_checks=8, distinct_training_views=8,
        optimizer_updates=0, parameters_unchanged=True, frozen_models_verified=True,
        current_protocol_source_input_identities_verified=True, immutable_raw_receipts_verified=True,
        exact_reviewed_target_support_and_provenance_verified=True,
        shared_view_seed_schedules_verified=True, historical_gradient_comparisons_recomputed=True,
        historical_source_input_pin_count=len(historical_identity['files']),
        candidate_source_input_pin_count=len(candidate_identity['files']),
        historical_identity_sha256=old_state['identity_hash'], candidate_identity_sha256=new_state['identity_hash'],
        historical_aggregate_sha256=object_hash(old_aggregate), candidate_aggregate_sha256=object_hash(new_aggregate),
        controlled_case_identity_sha256=original_cases['frozen_case_identity_sha256'],
        controlled_reports_reproduced_exactly=True, controlled_case_count=52,
        controlled_source_hashes_sha256=object_hash(dict(historical=original_cases['source_hashes'], surface=surface_cases['source_hashes'])),
        project_locks_absent_and_workers_exited=True, candidate_policy=POLICY,
        interpretation=interpretation(old_aggregate, new_aggregate, surface_cases),
        training_clearance=False, method_selected=False, label_accuracy_verified=False,
        server_gate_eligible=False, held_out_generalization_evidence=False)
    require(finite_tree((old_aggregate, new_aggregate, audit)), 'Nonfinite completion aggregates.')
    output.mkdir(parents=True)
    save_json(output / 'historical-aggregate.json', old_aggregate)
    save_json(output / 'candidate-aggregate.json', new_aggregate)
    save_json(output / 'completion-audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.output)
    print({k: result[k] for k in ('total_finite_checks', 'optimizer_updates', 'training_clearance')})
