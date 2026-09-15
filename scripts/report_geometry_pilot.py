"""Rebuild aggregate-only reports from complete, matched field pilot records."""
import argparse
import math
from pathlib import Path

from satground.common import ResearchError, load_json, read_jsonl, save_json, sha256, utc_now
from satground.metrics import grouped_mean, paired_group_bootstrap

METRICS = ('boundary_error', 'building_iou', 'lpips', 'ssim')


def read_evaluation(folder):
    folder = Path(folder)
    summary = load_json(folder / 'summary.json')
    rows = read_jsonl(folder / 'metrics.jsonl')
    if len(rows) != summary['count'] or not rows or any(r['split'] != 'validation' for r in rows):
        raise ResearchError('Complete validation evaluations are required.')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ResearchError('Duplicate sample IDs in evaluation.')
    generation = summary['generation']
    if not generation.get('scientific') or generation.get('target_images_used_for_conditioning') is not False:
        raise ResearchError('Real, target-free generation is required.')
    if generation['manifest_sha256'] != summary['manifest_sha256']:
        raise ResearchError('Generation and evaluation manifests differ.')
    for key in METRICS:
        if not math.isclose(grouped_mean(rows, key), summary['group_weighted'][key], abs_tol=1e-12):
            raise ResearchError('Evaluation aggregate differs from its per-view records.')
    return summary, rows


def require_matched(a, b):
    for key in ('manifest_sha256', 'segmenter', 'segmenter_revision'):
        if a[key] != b[key]:
            raise ResearchError('Evaluation input or segmenter mismatch: ' + key)
    ga, gb = a['generation'], b['generation']
    for key in ('style_sha256', 'train_manifest_sha256', 'train_data_digest', 'trained_steps'):
        if ga[key] != gb[key]:
            raise ResearchError('Training/style mismatch: ' + key)
    if ga.get('sampling') != gb.get('sampling'):
        raise ResearchError('Importance sampling conventions differ.')
    for key in ('pipeline_source_hash', 'model_revision', 'code_revision', 'packages'):
        if ga['provenance'][key] != gb['provenance'][key]:
            raise ResearchError('Generation source/model mismatch: ' + key)


def comparison(a, b):
    baseline = grouped_mean(a, 'boundary_error')
    lpips = grouped_mean(a, 'lpips')
    if baseline <= 0 or lpips <= 0:
        raise ResearchError('Relative changes require nonzero reference errors.')
    return dict(relative_boundary_reduction=(baseline - grouped_mean(b, 'boundary_error')) / baseline,
        boundary_bootstrap=paired_group_bootstrap(a, b),
        iou_change=grouped_mean(b, 'building_iou') - grouped_mean(a, 'building_iou'),
        relative_lpips_change=(grouped_mean(b, 'lpips') - lpips) / lpips)


def diagnostic(folder):
    folder = Path(folder)
    record = load_json(folder / 'summary.json')
    if record.get('status') != 'complete' or record.get('geometry_control_checks_passed') is not True:
        raise ResearchError('Field control checks are incomplete.')
    effects = read_jsonl(folder / 'effects.jsonl')
    if record['effects_sha256'] != sha256(folder / 'effects.jsonl') or len(effects) != record['count']:
        raise ResearchError('Field diagnostic records changed or are incomplete.')
    summaries, rows = {}, {}
    for mode in ('base', 'density', 'appearance', 'joint'):
        summaries[mode], rows[mode] = read_evaluation(folder / f'{mode}-eval')
        gen = summaries[mode]['generation']
        if gen.get('field_intervention') != mode or gen.get('sampling') != 'frozen_A0_coarse_plus_importance':
            raise ResearchError('Incorrect intervention or sampling convention.')
        if gen['checkpoint_sha256'] != record['checkpoint_sha256']:
            raise ResearchError('Diagnostic checkpoint mismatch.')
        require_matched(summaries['base'], summaries[mode])
        if {r['sample_id'] for r in rows[mode]} != {r['sample_id'] for r in effects}:
            raise ResearchError('Diagnostic and evaluation sample IDs differ.')
    keys = [k for k in effects[0] if k.endswith('_mae_from_base') or k == 'native_vs_shared_joint_rgb_mae']
    return dict(count=record['count'], geographic_groups=len({r['geo_group'] for r in effects}),
        checkpoint_sha256=record['checkpoint_sha256'], pipeline_source_hash=record['provenance']['pipeline_source_hash'],
        modes={m: summaries[m]['group_weighted'] for m in summaries},
        vs_base={m: comparison(rows['base'], rows[m]) for m in ('density', 'appearance', 'joint')},
        group_weighted_effects={k: grouped_mean(effects, k) for k in keys},
        native_baseline_max_rgb_difference=record['native_baseline_max_rgb_difference'],
        geometry_control_checks_passed=True, geometry_is_ground_truth=False,
        peak_vram_mib=record['peak_vram_mib'], elapsed_seconds=record['elapsed_seconds'],
        evidence_sha256={str((folder / p).as_posix()): sha256(folder / p)
                         for p in ('summary.json', 'effects.jsonl', 'protocol.json')})


def training_comparison(root):
    summaries, rows, training = {}, {}, {}
    for name, mode in [('GJ', 'joint'), ('GD', 'density')]:
        directory = root / f'geometry100-{name}-seed17'
        train = load_json(directory / 'summary.json')
        run = load_json(directory / 'run.json')
        log = read_jsonl(directory / 'training.jsonl')
        summaries[name], rows[name] = read_evaluation(root / f'geometry100-{name}-validation-eval')
        gen = summaries[name]['generation']
        cfg = gen['config']
        if cfg['experiment'] != name or cfg.get('field_mode') != mode:
            raise ResearchError('Incorrect training variant or field routing.')
        if train['status'] != 'budget_complete' or train['step'] != 100 or gen['trained_steps'] != 100:
            raise ResearchError('Both preliminary training budgets must be complete.')
        if [r['step'] for r in log] != list(range(1, 101)):
            raise ResearchError('Missing or repeated optimizer steps.')
        if any(not math.isfinite(v) for row in log for v in row.values() if isinstance(v, (int, float))):
            raise ResearchError('Nonfinite training values.')
        if gen['checkpoint_sha256'] != train['checkpoint_sha256'] or sha256(directory / 'last.pt') != train['checkpoint_sha256']:
            raise ResearchError('Generation did not use the complete verified checkpoint.')
        if gen['training_identity_hash'] != run['identity_hash'] or cfg != run['identity']['config']:
            raise ResearchError('Training identity or configuration mismatch.')
        if gen['training_provenance']['pipeline_source_hash'] != gen['provenance']['pipeline_source_hash']:
            raise ResearchError('Pilot source changed between training and generation.')
        training[name] = dict(**train, metrics=summaries[name]['group_weighted'], first_step=log[0],
            pipeline_source_hash=gen['provenance']['pipeline_source_hash'], training_identity_hash=gen['training_identity_hash'])
    require_matched(summaries['GJ'], summaries['GD'])
    a, b = [summaries[n]['generation']['config'] for n in ('GJ', 'GD')]
    if {k: v for k, v in a.items() if k not in ('experiment', 'field_mode')} != {
        k: v for k, v in b.items() if k not in ('experiment', 'field_mode')}:
        raise ResearchError('Geometry and joint controls have unmatched budgets or settings.')
    for key in ('rgb', 'lpips', 'region', 'boundary', 'total'):
        if training['GJ']['first_step'][key] != training['GD']['first_step'][key]:
            raise ResearchError('Initial zero-adapter losses do not match.')
    return dict(runs=training, GD_vs_GJ=comparison(rows['GJ'], rows['GD']),
                first_step_losses_identical=True, sample_count=len(rows['GD']))


def report(root, output, diagnostics_only=False):
    root, output = Path(root), Path(output)
    result = dict(created_utc=utc_now(), stage='preliminary_single_seed_development_probe',
        diagnostics={name: diagnostic(root / f'fields-{name}-seed17') for name in ('A1', 'A3')},
        human_review_pending=True, manually_verified_masks=0, building_identity_method_implemented=False,
        server_decision='insufficient_evidence', audit_used=False, Seattle_used=False,
        inference='one overhead RGB plus camera and fixed training illumination',
        limitations=['24 repeatedly inspected validation views; 10 geographic groups; one seed',
                     'pseudo-mask scores; no independently measured geometric accuracy',
                     'frozen proposal can miss newly occupied space',
                     'exploratory bootstrap intervals; no multiple-comparison correction'])
    if not diagnostics_only:
        result['training'] = training_comparison(root)
    lines = ['# Laptop geometry probe', '', 'Preliminary, single-seed development evidence. '
             'No server-scale or publication claim is established. Human review is pending.', '',
             '## Existing-adapter field interventions', '',
             'Each row uses the same frozen-baseline ray samples. Lower boundary/LPIPS and higher IoU are better.', '',
             '| Adapter | Field intervention | Boundary error | Building IoU | LPIPS |',
             '|---|---|---:|---:|---:|']
    for name, diagnostic_result in result['diagnostics'].items():
        for mode, values in diagnostic_result['modes'].items():
            lines.append(f"| {name} | {mode} | {values['boundary_error']:.6f} | {values['building_iou']:.6f} | {values['lpips']:.6f} |")
    lines += ['', 'Unchanged opacity/radial maps in the appearance-only intervention are a software control. '
              'Changes in the density intervention do not establish more accurate buildings. '
              'Raw maps and every preselected view remain in the ignored local runs directory.', '']
    if 'training' in result:
        lines += ['## Matched 100-step control', '', '| Run | Boundary error | Building IoU | LPIPS | Loop seconds | Peak allocated MiB |',
                  '|---|---:|---:|---:|---:|---:|']
        for name, values in result['training']['runs'].items():
            m = values['metrics']
            lines.append(f"| {name} | {m['boundary_error']:.6f} | {m['building_iou']:.6f} | {m['lpips']:.6f} | {values['elapsed_seconds']:.1f} | {values['peak_vram_mib']:.1f} |")
        c = result['training']['GD_vs_GJ']
        ci = c['boundary_bootstrap']['ci95']
        lines += ['', f"GD versus GJ: boundary reduction **{c['relative_boundary_reduction']:.2%}**; paired geographic-bootstrap "
                  f"95% interval for absolute improvement [{ci[0]:.6f}, {ci[1]:.6f}]. "
                  f"IoU change {100*c['iou_change']:.2f} percentage points; LPIPS change {c['relative_lpips_change']:.2%}.", '',
                  'Both use the same A3 image-space losses, adapter capacity, samples, and optimizer budget. '
                  'GJ adapts density and color features; GD uses baseline color features. '
                  'All feature caches were already populated. Loop memory excludes initialization and separate desktop/driver allocations.', '']
    lines += ['## Interpretation limits', '', *['- ' + item + '.' for item in result['limitations']], '',
              'The next method stage needs reviewed instance correspondences and a direct semantic-rendering control. '
              'These interventions and GD alone are not the proposed building-identity contribution.', '',
              'Protocol and commands: [geometry pilot](../../docs/GEOMETRY_PILOT.md). '
              'Full aggregate records: [summary.json](summary.json).', '']
    save_json(output / 'summary.json', result)
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--output', default='reports/geometry-pilot')
    parser.add_argument('--diagnostics-only', action='store_true')
    args = parser.parse_args()
    report(args.runs, args.output, args.diagnostics_only)
    print('Verified aggregate report written to', args.output)
