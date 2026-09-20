"""Inspect research prerequisites and retain verified historical trial evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path

from satground.common import ROOT, ResearchError, load_json, save_json, sha256, utc_now


SUMMARY = 'reports/contour500/comparison/summary.json'
AUDIT = 'reports/contour500/completion-audit.json'
PRIMARY = 'A3_corrected_vs_A2_fresh'
STRATA = {'all24': None, 'reviewed21': 'reviewed', 'target_building': 'target_building'}


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def finite_number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Expected finite numeric evidence.')
    return value


def inspect_upstream(root=ROOT):
    pin = load_json(root / 'configs/autoresearch/upstream.json')
    checkout = root / pin['local_path']
    command = ['git', '-C', str(checkout)]
    revision = subprocess.check_output([*command, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output([*command, 'status', '--porcelain'], text=True).strip()
    require(revision == pin['revision'] and not dirty, 'Imported autoresearch checkout changed.')
    for name, expected in pin['git_blob_sha256'].items():
        content = subprocess.check_output([*command, 'show', 'HEAD:' + name])
        require(hashlib.sha256(content).hexdigest() == expected, 'Imported reference hash differs.')
    return {'revision': revision, 'verified_files': len(pin['git_blob_sha256']),
            'training_executed': False, 'dependencies_installed': False}


def review_progress(root=ROOT):
    from training_review_store import TrainingReviewStore
    preparation = load_json(root / 'reports/TRAINING_REVIEW_STATUS.json')
    packet = root / 'data/review/training12-v1'
    require(sha256(packet / 'packet.json') == preparation['packet_sha256'], 'Training packet changed.')
    require(sha256(packet / 'manifest.jsonl') == preparation['manifest_sha256'], 'Training manifest changed.')
    store = TrainingReviewStore(packet, root / 'data/review/training12-edits')
    batch = store.batch_state()
    require(batch['total'] == preparation['training_views'] == 12, 'Training review cohort changed.')
    counts = {'pending': 0, 'supported': 0, 'uncertain': 0, 'mismatch': 0}
    for view in batch['views']:
        decision = store.review_state(view['number'])
        counts[decision['pairing']] += 1
    return {'total': batch['total'], 'finished': batch['ready'] + batch['accepted'],
            'remaining': batch['remaining'], 'pairing_counts': counts,
            'human_acceptance_verified_by_this_command': False}


def status(root=ROOT):
    progress = review_progress(root)
    return {'created_utc': utc_now(), 'upstream': inspect_upstream(root), 'review': progress,
            'state': 'awaiting_training_review' if progress['remaining'] else 'requires_review_verification',
            'new_training_started': False, 'training_launcher_implemented': False,
            'next_step': ('Complete the 12-view review, then verify exact masks and pairing evidence.' if progress['remaining'] else
                          'Resolve content follow-ups and freeze the eligible subset before specifying the bounded fitting test.'),
            'historical_experiment': 'contour500_complete_inconclusive', 'server_gate_eligible': False}


def assess_contour(summary, audit):
    require(summary['primary'] == PRIMARY and summary['fixed_final_step'] == 500 and summary['seed'] == 17,
            'Unexpected historical comparison or budget.')
    require(summary['validation_views'] == 24 and summary['reviewed_views'] == 21 and summary['pseudo_label_views'] == 3,
            'Unexpected historical evaluation cohort.')
    require(audit['status'] == 'verified' and audit['report_reproduced'] is True and
            audit['total_retained_steps'] == 1000, 'Historical audit is incomplete.')
    require(summary['manual_index_sha256'] == audit['manual_index_sha256'] and
            summary['runner_identity'] == audit['source_hash'], 'Historical evidence identity differs.')
    for model in ('A2', 'A3'):
        training = audit['training'][model]
        require(training['steps'] == 500 and training['all_values_finite'] is True, 'Invalid historical training.')
        require(finite_number(training['elapsed_seconds']) > 0 and finite_number(training['peak_vram_mib']) > 0,
                'Missing historical resource cost.')
    contrasts = summary['comparisons'][PRIMARY]
    checked = {}
    for name, stratum in STRATA.items():
        control = summary['models']['A2_fresh']
        candidate = summary['models']['A3_corrected']
        if stratum:
            control, candidate = control['strata'][stratum], candidate['strata'][stratum]
            require(control['views'] == candidate['views'], 'Stratum cohort differs.')
        reference, proposed = control['metrics'], candidate['metrics']
        for metrics in (reference, proposed):
            for key in ('boundary_error', 'building_iou', 'lpips'):
                require(finite_number(metrics[key]) >= 0, 'Negative metric.')
            require(metrics['boundary_error'] <= 1 and metrics['building_iou'] <= 1, 'Metric outside its domain.')
        require(reference['boundary_error'] > 0 and reference['lpips'] > 0, 'Undefined relative comparison.')
        computed = {'relative_boundary_reduction': 1 - proposed['boundary_error'] / reference['boundary_error'],
                    'building_iou_change': proposed['building_iou'] - reference['building_iou'],
                    'relative_lpips_change': proposed['lpips'] / reference['lpips'] - 1}
        for key, value in computed.items():
            require(math.isclose(value, finite_number(contrasts[name][key]), rel_tol=1e-9, abs_tol=1e-12),
                    'Aggregate comparison disagrees with its model metrics.')
        bootstrap = contrasts[name]['boundary_bootstrap']
        interval = bootstrap['ci95']
        require(len(interval) == 2 and finite_number(interval[0]) <= finite_number(interval[1]), 'Invalid interval.')
        require(bootstrap['weighting'] == 'equal_geographic_group' and bootstrap['geographic_groups'] > 1,
                'Geographic interval is missing.')
        expected_count = 24 if name == 'all24' else control['views']
        require(bootstrap['samples'] == expected_count, 'Interval cohort differs.')
        checked[name] = {**computed, 'boundary_improvement_ci95': interval}
    empty_control = summary['models']['A2_fresh']['strata']['target_empty']
    empty_candidate = summary['models']['A3_corrected']['strata']['target_empty']
    require(empty_control['views'] == empty_candidate['views'] == 3 and
            empty_control['valid_pixels'] == empty_candidate['valid_pixels'] > 0, 'Empty-target cohort differs.')
    empty_change = finite_number(empty_candidate['group_weighted_false_positive_fraction']) - finite_number(
        empty_control['group_weighted_false_positive_fraction'])
    quality_ok = all(result['building_iou_change'] >= -0.01 and result['relative_lpips_change'] <= 0.05
                     for result in checked.values()) and empty_change <= 0
    direction_supported = all(result['relative_boundary_reduction'] > 0 and result['boundary_improvement_ci95'][0] > 0
                              for result in checked.values())
    decision = 'inconclusive'
    if not quality_ok:
        decision = 'quality_regression'
    elif direction_supported:
        decision = 'exploratory_signal_requires_independent_confirmation'
    return {'trial': 'historical-contour500-seed17', 'kind': 'historical_import_no_new_training',
            'comparison': PRIMARY, 'steps_per_model': 500, 'seed': 17,
            'measured_contrasts': checked, 'empty_false_positive_fraction_change': empty_change,
            'quality_constraints_passed': quality_ok, 'geographic_intervals_support_direction': direction_supported,
            'decision': decision, 'candidate_promoted': False, 'server_gate_eligible': False,
            'training_resources': audit['training'],
            'limitations': ['One seed; already inspected development views; incomplete data QA.',
                            '21 reviewed evaluation masks and three pseudo-label targets.',
                            'Historical A2 retraining was not bitwise reproducible.',
                            'No novelty, audited generalization or server-gate claim.']}


def ledger_tsv(record):
    output = io.StringIO(newline='')
    writer = csv.writer(output, delimiter='\t', lineterminator='\n')
    writer.writerow(['trial', 'comparison', 'steps_per_model', 'seed', 'reviewed_boundary_reduction',
                     'reviewed_ci_low', 'reviewed_ci_high', 'decision', 'candidate_promoted'])
    reviewed = record['measured_contrasts']['reviewed21']
    writer.writerow([record['trial'], record['comparison'], record['steps_per_model'], record['seed'],
                     reviewed['relative_boundary_reduction'], *reviewed['boundary_improvement_ci95'],
                     record['decision'], str(record['candidate_promoted']).lower()])
    return output.getvalue()


def local_session(path, root=ROOT):
    resolved = (root / path).resolve()
    allowed = (root / 'runs/autoresearch').resolve()
    require(resolved != allowed and resolved.is_relative_to(allowed), 'Use a session beneath runs/autoresearch/.')
    return resolved


def import_contour(output):
    from audit_contour_completion import audit as audit_completed_contour
    output = local_session(output)
    require(not output.exists(), 'Use a fresh session; previous results are immutable.')
    upstream = inspect_upstream()
    output.mkdir(parents=True)
    audit_completed_contour(output / 'completion-audit.json')
    original, repeated = load_json(ROOT / AUDIT), load_json(output / 'completion-audit.json')
    require({key: value for key, value in original.items() if key != 'created_utc'} ==
            {key: value for key, value in repeated.items() if key != 'created_utc'}, 'Fresh audit differs from preserved evidence.')
    record = assess_contour(load_json(ROOT / SUMMARY), repeated)
    record['source_sha256'] = {SUMMARY: sha256(ROOT / SUMMARY), AUDIT: sha256(ROOT / AUDIT)}
    save_json(output / 'record.json', record)
    (output / 'results.tsv').write_text(ledger_tsv(record), encoding='utf-8', newline='')
    session = {'created_utc': utc_now(), 'upstream': upstream, 'new_training_updates': 0,
               'files': {name: sha256(output / name) for name in ('record.json', 'results.tsv', 'completion-audit.json')},
               'tool_sha256': sha256(__file__)}
    save_json(output / 'session.json', session)
    return {'decision': record['decision'], 'new_training_updates': 0, 'session_verified': verify_ledger(output)}


def verify_ledger(output, root=ROOT):
    output = local_session(output, root)
    session = load_json(output / 'session.json')
    require(set(session['files']) == {'record.json', 'results.tsv', 'completion-audit.json'}, 'Ledger file set changed.')
    for name, digest in session['files'].items():
        require(sha256(output / name) == digest, 'Ledger artifact changed.')
    record = load_json(output / 'record.json')
    require(set(record['source_sha256']) == {SUMMARY, AUDIT}, 'Ledger source set changed.')
    for name, digest in record['source_sha256'].items():
        require(sha256(root / name) == digest, 'Historical evidence changed.')
    original = load_json(root / AUDIT)
    repeated = load_json(output / 'completion-audit.json')
    require({key: value for key, value in original.items() if key != 'created_utc'} ==
            {key: value for key, value in repeated.items() if key != 'created_utc'}, 'Ledger audit differs from preserved evidence.')
    expected = assess_contour(load_json(root / SUMMARY), repeated)
    require({key: value for key, value in record.items() if key != 'source_sha256'} == expected,
            'Ledger decision does not reproduce.')
    require((output / 'results.tsv').read_text(encoding='utf-8') == ledger_tsv(record), 'TSV does not reproduce.')
    require(session['new_training_updates'] == 0, 'Historical import cannot train.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('status')
    importer = commands.add_parser('import-contour')
    importer.add_argument('--output', required=True)
    verifier = commands.add_parser('verify-ledger')
    verifier.add_argument('--session', required=True)
    args = parser.parse_args()
    if args.command == 'status':
        result = status()
    elif args.command == 'import-contour':
        result = import_contour(args.output)
    else:
        result = {'verified': verify_ledger(args.session)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
