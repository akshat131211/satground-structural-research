"""Freeze a conditional v2 trial without copying or changing reviewed labels."""
import argparse
from pathlib import Path

import numpy as np
import yaml

import probe_saturated_surface_gradients as feasibility
from prepare_reviewed_fitting import check_pins, require, verify_bundle
from saturated_surface_objective import POLICY, prepare_surface_labels
from satground.common import ROOT, load_json, object_hash, provenance, read_jsonl, save_json, sha256, utc_now
from satground.runs import read_config


PROTOCOL = ROOT / 'configs/saturated-fitting200/protocol.json'
PROTOCOL_DOC = ROOT / 'docs/VALID_SURFACE_SATURATION_REVISION.md'
ORIGINAL_BUNDLE = ROOT / 'data/review/reviewed-fitting8-v1'
OUTPUT = ROOT / 'data/review/saturated-fitting8-v2'


def validate_protocol(protocol):
    expected = dict(study='reviewed_saturated_surface_fitting', original_bundle='data/review/reviewed-fitting8-v1',
        completed_feasibility='runs/reviewed-saturated-surface8',
        controlled_report='reports/saturated-surface-diagnosis/controlled-cases.json',
        steps=200, seed=17, render_seed=101, learning_rate=.00003, weight_decay=.0001, size=256,
        adapter_width=16, chunk_rows=4, checkpoint_rays=True, amp=False, accumulation=8, save_every=50,
        perceptual_weight=.1, region_weight=.5, boundary_weight=0., a3_surface_weight=.5,
        surface_policy=POLICY, radius=8, class_mass_floor=32,
        training_segmenter_revision='21b3847fae21ddee674abd31129307b6a1235bd9',
        evaluation_segmenter_revision='ec86afeba68e656629ccf47e0c8d2902f964917b',
        decision=dict(minimum_relative_boundary_reduction=.1, minimum_relative_nonperimeter_reduction=.1,
            minimum_iou_change=-.01, maximum_relative_lpips_increase=.05, minimum_fraction_views_jointly_improved=.5),
        startup_resume_updates_included_in_budget=1, main_study_eligible=False,
        server_gate_eligible=False, held_out_generalization_evidence=False)
    require(protocol == expected, 'Only the fixed saturated 200-step seed-17 protocol is supported.')


def make_configs(trial_dir, protocol, original_bundle):
    validate_protocol(protocol)
    original_bundle, trial_dir = Path(original_bundle).resolve(), Path(trial_dir).resolve()
    result = {}
    for name in ('A0', 'A2', 'A3'):
        cfg = read_config(original_bundle / f'{name}.yaml')
        for key in ('steps', 'seed', 'render_seed', 'learning_rate', 'weight_decay', 'size', 'adapter_width',
                    'chunk_rows', 'checkpoint_rays', 'amp', 'accumulation', 'save_every', 'perceptual_weight',
                    'training_segmenter_revision'):
            require(cfg[key] == protocol[key], 'Original fitting renderer/budget differs: ' + key)
        require(Path(cfg['fitting_bundle']).resolve() == original_bundle
                and Path(cfg['fitting_manifest']).resolve() == original_bundle / 'manifest.jsonl'
                and cfg['boundary_mode'] == 'legacy' and cfg['region_weight'] == (0. if name == 'A0' else .5),
                'Original fitting labels/region settings differ.')
        cfg.update(boundary_weight=0., surface_weight=.5 if name == 'A3' else 0.,
                   surface_policy=POLICY, saturated_trial=str(trial_dir))
        result[name] = cfg
    require({k: v for k, v in result['A2'].items() if k not in ('experiment', 'surface_weight')} ==
            {k: v for k, v in result['A3'].items() if k not in ('experiment', 'surface_weight')},
            'A2/A3 differ beyond the declared surface coefficient.')
    return result


def source_pins():
    paths = [PROTOCOL, PROTOCOL_DOC, ROOT / 'program.md', *[ROOT / 'scripts' / name for name in
        ('prepare_saturated_fitting.py', 'train_saturated_fitting.py', 'run_saturated_fitting.py',
         'report_saturated_fitting.py', 'saturated_surface_objective.py', 'valid_surface_objective.py',
         'audit_reviewed_fitting_outputs.py')],
        ROOT / 'artifacts/asset-lock.json', ROOT / 'artifacts/Sat3DGen/config.json',
        ROOT / 'artifacts/Sat3DGen/diffusion_pytorch_model.safetensors',
        *sorted((ROOT / 'src/satground').glob('*.py')),
        *sorted((ROOT / 'external/Sat3DGen/source').rglob('*.py'))]
    # Resolve existing frozen reference weights offline; preparation never
    # downloads a model or changes shared cache contents.
    from huggingface_hub import try_to_load_from_cache
    from satground.semantics import TRAIN_SEGMENTER, EVAL_SEGMENTER
    import importlib.util
    import torch
    protocol = load_json(PROTOCOL)
    for model, revision in ((TRAIN_SEGMENTER, protocol['training_segmenter_revision']),
                            (EVAL_SEGMENTER, protocol['evaluation_segmenter_revision'])):
        for filename in ('config.json', 'pytorch_model.bin'):
            cached = try_to_load_from_cache(model, filename, revision=revision)
            require(isinstance(cached, str), 'Missing existing frozen semantic reference cache.')
            paths.append(Path(cached))
    lpips_spec = importlib.util.find_spec('lpips')
    require(lpips_spec is not None and lpips_spec.origin is not None, 'Missing frozen perceptual reference package.')
    paths.extend((Path(lpips_spec.origin).parent / 'weights/v0.1/alex.pth',
                  Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'))
    require(all(path.is_file() for path in paths), 'Missing saturated trial source; finish implementation before freezing.')
    return {str(path.resolve()): sha256(path) for path in paths}


def verified_prerequisites(protocol):
    _, bundle, _, rows, _, _, identity = feasibility.protocol_identity(feasibility.PROTOCOL)
    output = ROOT / protocol['completed_feasibility']
    state = load_json(output / 'status.json')
    require(state['status'] == 'complete' and state.get('current_check') is None
            and state['completed'] == state['expected'] == 8 and state['optimizer_updates'] == 0
            and state['parameters_unchanged'] is True and state['identity'] == identity
            and state['identity_hash'] == object_hash(identity), 'Saturated feasibility is incomplete or incompatible.')
    records = feasibility.verified_records(output, state)
    summary = feasibility.aggregate(records, state)
    require(load_json(output / 'summary.json') == summary and sha256(output / 'summary.json') == state['summary_sha256']
            and summary['feasibility_rail_passed'] is True
            and summary['supported_nonzero_finite_adapter_views'] >= 6
            and summary['controlled_prerequisite_clearance'] is True,
            'Saturated feasibility rail has not passed; no trial may be prepared.')
    require(Path(bundle).resolve() == ORIGINAL_BUNDLE.resolve() and len(rows) == 8, 'Different reviewed cohort.')
    for row, record in zip(rows, records):
        with np.load(Path(bundle) / 'labels' / (row['sample_id'] + '.npz'), allow_pickle=False) as values:
            prepared = prepare_surface_labels(values['building'], values['valid'])
        require(prepared['counts'] == record['support'] and prepared['provenance'] == record['surface_provenance'],
                'Actual reviewed masks do not support the saturated feasibility record.')
    pins = dict(identity['files'])
    for file in (output / 'status.json', output / 'summary.json', *sorted((output / 'checks').glob('*/*'))):
        require(file.is_file(), 'Unexpected feasibility evidence path.')
        pins[str(file.resolve())] = sha256(file)
    check_pins(pins)
    return pins, state['identity_hash'], state['summary_sha256']


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    require(output == OUTPUT.resolve() and not output.exists(), 'Use the fresh fixed private saturated trial path.')
    protocol = load_json(PROTOCOL)
    validate_protocol(protocol)
    original = verify_bundle(ORIGINAL_BUNDLE)
    require(original['fitting_count'] == 8, 'The trial requires the unchanged eight fitting views.')
    pins, feasibility_identity, feasibility_summary = verified_prerequisites(protocol)
    pins.update(original['source_file_sha256'])
    pins.update(original['artifact_sha256'])
    pins.update(source_pins())
    pins[str((ORIGINAL_BUNDLE / 'bundle.json').resolve())] = sha256(ORIGINAL_BUNDLE / 'bundle.json')
    configs = make_configs(output, protocol, ORIGINAL_BUNDLE)
    check_pins(pins)
    output.mkdir(parents=True)
    for name, config in configs.items():
        (output / f'{name}.yaml').write_text(yaml.safe_dump(config, sort_keys=True), encoding='utf-8')
    record = dict(kind='saturated_surface_training_fitting_trial', scope='training_fitting_only',
        created_utc=utc_now(), protocol=protocol, original_bundle=str(ORIGINAL_BUNDLE.resolve()),
        original_bundle_sha256=sha256(ORIGINAL_BUNDLE / 'bundle.json'),
        source_file_sha256=pins, artifact_sha256={str(p.resolve()): sha256(p) for p in sorted(output.glob('*.yaml'))},
        pipeline_source_hash=provenance()['pipeline_source_hash'], fitting_count=8,
        feasibility_identity_sha256=feasibility_identity, feasibility_summary_sha256=feasibility_summary,
        original_masks_copied=False, human_submitted_masks_modified=False, main_study_eligible=False,
        server_gate_eligible=False, held_out_generalization_evidence=False)
    save_json(output / 'trial.json', record)
    verify_trial(output)
    return record


def verify_trial(trial_dir):
    trial_dir = Path(trial_dir).resolve()
    record = load_json(trial_dir / 'trial.json')
    require(record['kind'] == 'saturated_surface_training_fitting_trial' and record['scope'] == 'training_fitting_only'
            and record['original_masks_copied'] is False and record['human_submitted_masks_modified'] is False
            and record['main_study_eligible'] is False and record['fitting_count'] == 8,
            'Invalid saturated trial provenance.')
    validate_protocol(record['protocol'])
    require(record['protocol'] == load_json(PROTOCOL), 'Trial protocol changed.')
    original = Path(record['original_bundle']).resolve()
    require(original == ORIGINAL_BUNDLE.resolve() and sha256(original / 'bundle.json') == record['original_bundle_sha256'],
            'The original reviewed bundle changed.')
    require(verify_bundle(original)['fitting_count'] == 8, 'Wrong original fitting cohort.')
    check_pins(record['source_file_sha256'])
    check_pins(record['artifact_sha256'])
    require(set(record['artifact_sha256']) == {str(trial_dir / f'{name}.yaml') for name in ('A0', 'A2', 'A3')}
            and {p.name for p in trial_dir.iterdir()} == {'trial.json', 'A0.yaml', 'A2.yaml', 'A3.yaml'},
            'Trial contains unexpected copied labels or configuration artifacts.')
    configs = make_configs(trial_dir, record['protocol'], original)
    require(all(read_config(trial_dir / f'{name}.yaml') == cfg for name, cfg in configs.items()),
            'Saturated trial configurations differ from the frozen contract.')
    require(provenance()['pipeline_source_hash'] == record['pipeline_source_hash'], 'Core source changed.')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(OUTPUT))
    args = parser.parse_args()
    result = prepare(args.output)
    print({k: result[k] for k in ('kind', 'fitting_count', 'human_submitted_masks_modified')})
