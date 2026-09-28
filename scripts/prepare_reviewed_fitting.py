"""Freeze an explicitly resolved training subset without modifying submitted masks."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from satground.annotations import verified_reviewed_masks
from satground.backend import load_style
from satground.camera import crop_panorama
from satground.common import (ROOT, ResearchError, load_json, object_hash, read_jsonl,
                             require_development, safe_data_path, save_json, sha256, utc_now, write_jsonl)
from verify_training_review import mask_measurements

PROTOCOL = ROOT / 'configs/reviewed-fitting200/protocol.json'
PROTOCOL_DOC = ROOT / 'docs/REVIEWED_FITTING.md'
PARENT = ROOT / 'data/manifests/learning100/train.jsonl'
DEVELOPMENT = ROOT / 'data/manifests/learning100/validation.jsonl'
STYLE = ROOT / 'data/style/learning100-train-mean.json'


def require(ok, message):
    if not ok:
        raise ResearchError(message)


def check_pins(pins):
    require(bool(pins), 'Missing pinned evidence.')
    for path, digest in pins.items():
        require(Path(path).is_file() and sha256(path) == digest, 'Pinned fitting evidence changed: ' + str(path))


def selected_reviews(snapshot, decision):
    """An agent may record an actual answer, but this function never supplies one."""
    require(decision.get('snapshot_content_hash') == object_hash(snapshot), 'Disposition belongs to a different snapshot content.')
    require(decision.get('source') == 'user_message' and bool(str(decision.get('raw_user_answer', '')).strip())
            and bool(decision.get('recorded_utc')), 'Record the actual user disposition; pending is not a choice.')
    submissions = snapshot['submissions']
    numbers = [str(e['number']) for e in submissions]
    require(len(numbers) == len(set(numbers)), 'Duplicate submitted view.')
    dispositions = decision.get('dispositions', {})
    require(set(dispositions) == set(numbers), 'Every submitted view needs an explicit disposition.')
    selected = []
    for entry in submissions:
        disposition = dispositions[str(entry['number'])]
        require(disposition.get('action') in ('include', 'exclude') and disposition.get('reason'), 'Unresolved view disposition.')
        if disposition['action'] == 'exclude':
            continue
        review = entry['human_review']
        require(review['pairing'] == 'supported' and review['support'] == 'mostly_inside'
                and review['mask_review'] == 'checked', 'Uncertain or mismatched views cannot enter fitting.')
        require(entry['measurements']['interior_valid_boundary_pixels'] > 0, 'Included view has no usable target contour.')
        # The known content issue cannot be waived by a generic all-masks-ready reply.
        if entry['number'] == 1:
            resolution = disposition.get('content_resolution', {})
            require(resolution.get('action') == 'corrected_and_inspected'
                    and resolution.get('version') == entry['version']
                    and resolution.get('previous_version') != entry['version']
                    and bool(resolution.get('previous_version')) and bool(resolution.get('inspection')),
                    'View 1 needs an exact revised version and documented ground-label inspection, or exclusion.')
        selected.append(entry)
    require(bool(selected), 'No eligible views remain.')
    return sorted(selected, key=lambda e: e['number'])


def make_configs(bundle, protocol, parent=PARENT, style=STYLE):
    """Parent train_manifest is the histogram source; fitting_manifest alone drives updates."""
    require(protocol['steps'] == 200 and protocol['seed'] == 17 and protocol['learning_rate'] == .00003
            and protocol['save_every'] == 50 and protocol['amp'] is False,
            'Only the fixed 200-step, seed-17, float32 fitting budget is supported.')
    shared = {k: protocol[k] for k in ('seed', 'render_seed', 'size', 'adapter_width', 'chunk_rows',
              'checkpoint_rays', 'amp', 'accumulation', 'steps', 'save_every', 'learning_rate',
              'weight_decay', 'perceptual_weight')}
    shared.update(train_manifest=str(Path(parent).resolve()), style=str(Path(style).resolve()),
                  fitting_manifest=str(Path(bundle).resolve() / 'manifest.jsonl'),
                  fitting_bundle=str(Path(bundle).resolve()), feature_cache='.cache/features',
                  training_segmenter_revision=protocol['training_segmenter_revision'],
                  data_review_mode='exploratory', boundary_mode='legacy',
                  exploratory_reason='Training-example fitting diagnosis using resolved reviewed masks; independent data QA incomplete.')
    return {name: dict(shared, experiment=name, region_weight=0 if name == 'A0' else protocol['region_weight'],
                       boundary_weight=protocol['a3_boundary_weight'] if name == 'A3' else 0)
            for name in ('A0', 'A2', 'A3')}


def prepare(snapshot_path, decision_path, output, data_root, parent=PARENT, development=DEVELOPMENT,
            style=STYLE, protocol_path=PROTOCOL, expected_count=12):
    snapshot_path, output = Path(snapshot_path).resolve(), Path(output).resolve()
    require(output.is_relative_to(snapshot_path.parent.parent) and output != snapshot_path.parent,
            'Keep fitting bundles in the private review directory, separate from submissions.')
    require(not output.exists(), 'Use a fresh fitting bundle; existing evidence is immutable.')
    snapshot = load_json(snapshot_path)
    check_pins(snapshot['source_file_sha256'])
    require(len(snapshot['submissions']) == expected_count, 'Wrong source submission count.')
    selected = selected_reviews(snapshot, load_json(decision_path))
    protocol = load_json(protocol_path)
    parent_rows, dev_rows = read_jsonl(parent), read_jsonl(development)
    require_development(parent_rows, {'train'})
    require_development(dev_rows, {'validation'})
    by_id = {r['sample_id']: r for r in parent_rows}
    # Read metadata to prevent reuse; never open any development RGB or masks here.
    for entry in selected:
        require(entry['sample_id'] in by_id, 'Review outside the training universe.')
    rows = [by_id[e['sample_id']] for e in selected]
    for key in ('sample_id', 'satellite', 'panorama', 'geo_group'):
        require(not {r[key] for r in rows} & {r[key] for r in dev_rows}, 'Fitting/development overlap: ' + key)
        require(len({r[key] for r in rows}) == len(rows), 'Expected distinct fitting identities: ' + key)
    require(all(r['size'] == protocol['size'] for r in rows), 'Protocol resolution differs from fitting crops.')
    load_style(style, sha256(parent))
    pins = dict(snapshot['source_file_sha256'])
    for path in (snapshot_path, decision_path, parent, development, style, protocol_path, PROTOCOL_DOC):
        pins[str(Path(path).resolve())] = sha256(path)
    manual, arrays, supports = {}, {}, []
    for row, entry in zip(rows, selected):
        for key in ('satellite', 'panorama'):
            path = safe_data_path(data_root, row[key])
            pins[str(path)] = sha256(path)
        with Image.open(safe_data_path(data_root, row['panorama'])) as image:
            target = crop_panorama(image.convert('RGB'), row['yaw'], row['pitch'], row['fov'], row['size'])
        review = entry['human_review']
        reviewed = dict(reviewed=True, reviewer=review['reviewer'], reviewed_utc=entry['completed_utc'],
                        annotation_method='human_reviewed_corrected_training_proposal',
                        review_blinded_to_model_predictions=False, instance_errors=None,
                        submission_version=entry['version'], usage='training_fitting_only')
        for key in ('target_image', 'building_mask', 'valid_mask'):
            path = str(Path(entry['paths'][key]).resolve())
            require(path in pins, 'Submitted artifact lacks an original evidence hash.')
            reviewed[key], reviewed[key + '_sha256'] = path, pins[path]
        building, valid = verified_reviewed_masks(reviewed, target)
        support = mask_measurements(building, valid, building, valid)['interior_valid_boundary_pixels']
        require(support == entry['measurements']['interior_valid_boundary_pixels'] and support > 0,
                'Submitted contour support changed.')
        manual[row['sample_id']] = reviewed
        arrays[row['sample_id']] = {'building': building.astype(np.float32), 'valid': valid.astype(np.float32)}
        supports.append(support)
    check_pins(pins)
    output.mkdir()
    labels = output / 'labels'
    labels.mkdir()
    for sid, values in arrays.items():
        np.savez_compressed(labels / f'{sid}.npz', **values)
    write_jsonl(output / 'manifest.jsonl', rows)
    save_json(output / 'manual-index.json', manual)
    save_json(output / 'disposition.json', load_json(decision_path))
    for name, config in make_configs(output, protocol, parent, style).items():
        (output / f'{name}.yaml').write_text(yaml.safe_dump(config, sort_keys=True), encoding='utf-8')
    artifacts = [output / 'manifest.jsonl', output / 'manual-index.json', output / 'disposition.json',
                 *sorted(output.glob('*.yaml')), *sorted(labels.glob('*.npz'))]
    record = dict(kind='human_reviewed_training_masks', scope='training_fitting_only',
                  created_utc=utc_now(), source_file_sha256=pins,
                  artifact_sha256={str(p.resolve()): sha256(p) for p in artifacts},
                  training_segmenter_revision=protocol['training_segmenter_revision'],
                  protocol=protocol, selected_numbers=[e['number'] for e in selected],
                  submitted_count=len(snapshot['submissions']), fitting_count=len(rows),
                  excluded_count=len(snapshot['submissions']) - len(rows),
                  city_counts=dict(Counter(r['city'] for r in rows)),
                  positive_contour_pixels=sum(supports), source_snapshot_sha256=sha256(snapshot_path),
                  parent_training_manifest_sha256=sha256(parent), fitting_manifest_sha256=sha256(output / 'manifest.jsonl'),
                  human_submitted_masks_modified=False, main_study_eligible=False, sealed_images_opened=False)
    save_json(output / 'bundle.json', record)
    verify_bundle(output)
    return record


def verify_bundle(directory):
    directory = Path(directory).resolve()
    record = load_json(directory / 'bundle.json')
    require(record.get('kind') == 'human_reviewed_training_masks'
            and record.get('scope') == 'training_fitting_only'
            and record.get('human_submitted_masks_modified') is False
            and record.get('main_study_eligible') is False, 'Invalid fitting provenance.')
    check_pins(record['source_file_sha256'])
    check_pins(record['artifact_sha256'])
    require(all(Path(p).is_relative_to(directory) for p in record['artifact_sha256']), 'Bundle artifact escaped private directory.')
    rows = read_jsonl(directory / 'manifest.jsonl')
    require_development(rows, {'train'})
    require(sha256(directory / 'manifest.jsonl') == record['fitting_manifest_sha256'], 'Wrong fitting manifest.')
    manual = load_json(directory / 'manual-index.json')
    require(len(rows) == record['fitting_count'] and set(manual) == {r['sample_id'] for r in rows}, 'Incomplete reviewed cohort.')
    total = 0
    for row in rows:
        entry = manual[row['sample_id']]
        with Image.open(entry['target_image']) as image:
            building, valid = verified_reviewed_masks(entry, np.asarray(image.convert('RGB')))
        path = directory / 'labels' / f"{row['sample_id']}.npz"
        require(str(path) in record['artifact_sha256'], 'Unpinned fitting label.')
        with np.load(path, allow_pickle=False) as values:
            require(set(values.files) == {'building', 'valid'}, 'Unexpected reviewed-label channels.')
            for name, original in (('building', building), ('valid', valid)):
                require(values[name].dtype == np.float32 and np.array_equal(values[name], original.astype(np.float32)),
                        'Converted label differs from the submitted mask.')
        total += mask_measurements(building, valid, building, valid)['interior_valid_boundary_pixels']
    require(total == record['positive_contour_pixels'], 'Contour support total changed.')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='data/review/training12-submitted-v1/submissions.json')
    parser.add_argument('--decision', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='data/vigor')
    args = parser.parse_args()
    result = prepare(args.snapshot, args.decision, args.output, args.data_root)
    print({k: result[k] for k in ('kind', 'fitting_count', 'excluded_count', 'positive_contour_pixels')})
