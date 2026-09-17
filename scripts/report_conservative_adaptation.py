"""Verify region/boundary ablations alongside the unchanged A3/A1 primary comparison."""
import json
import math
from pathlib import Path

import torch

from audit_exploratory500 import finite_values, require
from report_checkpoint_diagnosis import strata, verify_evaluation
from report_exploratory_pair import report as paired_report
from satground.common import load_json, read_jsonl, sha256, utc_now
from satground.evaluation import comparison


def report(root, output):
    root, output = Path(root), Path(output)
    require(not output.exists(), 'Use fresh report output.')
    reference = load_json(root / 'A1-validation-eval/summary.json')
    reference_config = reference['generation']['config']
    methods, labels = {}, None
    for name in ('A1', 'A2', 'A3'):
        checkpoint = root / f'{name}-train/last.pt'
        s, rows = verify_evaluation(root / f'{name}-validation-eval', root / f'{name}-validation',
                                    reference, 500, checkpoint)
        g = s['generation']
        config = g['config']
        expected = {'A1': (0.0, 0.0), 'A2': (.5, 0.0), 'A3': (.5, .5)}[name]
        require(config['experiment'] == name and (config['region_weight'], config['boundary_weight']) == expected,
                'Unexpected experiment objectives.')
        require(config['learning_rate'] == .00003 and config['seed'] == 17 and
                config.get('data_review_mode') == 'exploratory', 'Unexpected revision settings.')
        allowed = {'experiment', 'region_weight', 'boundary_weight'}
        require({k: v for k, v in config.items() if k not in allowed} ==
                {k: v for k, v in reference_config.items() if k not in allowed}, 'Unmatched training resources.')
        for key in ('train_data_digest', 'training_readiness', 'training_provenance'):
            if key == 'training_provenance':
                for sub in ('pipeline_source_hash', 'code_revision', 'model_revision', 'packages'):
                    require(g[key][sub] == reference['generation'][key][sub], 'Training source mismatch.')
            else:
                require(g[key] == reference['generation'][key], 'Training data/readiness mismatch.')
        current = {r['sample_id']: tuple(r[k] for k in ('geo_group', 'label_provenance', 'valid_pixels', 'target_building_pixels')) for r in rows}
        if labels is None:
            labels = current
        require(current == labels, 'Scoring labels differ.')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(state['step'] == 500 and state['config'] == config, 'Checkpoint state/config mismatch.')
        finite = {k: finite_values(state[k]) for k in ('adapter', 'optimizer')}
        train = load_json(root / f'{name}-train/summary.json')
        require(train['status'] == 'budget_complete' and train['step'] == 500 and
                train['checkpoint_sha256'] == sha256(checkpoint), 'Training incomplete or checkpoint changed.')
        log = read_jsonl(root / f'{name}-train/training.jsonl')
        require([r['step'] for r in log] == list(range(1, 501)) and all(
            math.isfinite(v) for r in log for v in r.values() if isinstance(v, (float, int))), 'Invalid training log.')
        methods[name] = dict(metrics=s['group_weighted'], strata=strata(rows), checkpoint_sha256=sha256(checkpoint),
                             evaluation_sha256=sha256(root / f'{name}-validation-eval/summary.json'),
                             finite_tensor_values=finite, elapsed_seconds=train['elapsed_seconds'],
                             peak_vram_mib=train['peak_vram_mib'], trainable_parameters=train['trainable_parameters'])
    # The primary paired report performs additional A1/A3 provenance and integrity checks.
    paired_report(root, output / 'primary')
    ablations = {a + '_vs_' + b: comparison(root / f'{b}-validation-eval/metrics.jsonl',
                                          root / f'{a}-validation-eval/metrics.jsonl', 17)
                 for a, b in [('A3', 'A1'), ('A2', 'A1'), ('A3', 'A2')]}
    result = dict(created_utc=utc_now(), study='exploratory_pending_data_review', server_gate='not_eligible',
                  primary_comparison='A3_vs_A1', methods=methods, comparisons=ablations,
                  interpretation_pending=True, human_review_expansion_pending=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    lines = ['# Smaller-learning-rate comparison', '',
             'Fixed 500 steps, seed 17, learning rate 0.00003. Pending human review; no server decision.', '',
             '| Model | Boundary error | IoU | LPIPS |', '|---|---:|---:|---:|']
    for name, method in methods.items():
        m = method['metrics']
        lines.append(f"| {name} | {m['boundary_error']:.6f} | {m['building_iou']:.6f} | {m['lpips']:.6f} |")
    lines += ['', '[Primary comparison](primary/report.md). [Ablations and stratum diagnostics](summary.json).',
              'Interpret against frozen A0 and inspected reviewed views before any improvement claim.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return result
