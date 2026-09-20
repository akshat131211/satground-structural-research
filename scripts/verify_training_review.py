"""Freeze exact completed training reviews without assigning training eligibility."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

from satground.annotations import binary_mask
from satground.common import ResearchError, load_json, read_jsonl, require_development, save_json, sha256, utc_now
from satground.editor import unpack_mask
from training_review_store import TrainingReviewStore


def require(condition, message):
    if not condition:
        raise ResearchError(message)


def mask_measurements(building, valid, initial_building, initial_valid):
    edge = maximum_filter(building, size=3, mode='nearest') != minimum_filter(building, size=3, mode='nearest')
    interior = minimum_filter(valid, size=3, mode='nearest')
    return {'building_pixels': int(building.sum()), 'valid_pixels': int(valid.sum()),
            'scored_building_pixels': int((building & valid).sum()),
            'raw_boundary_pixels': int(edge.sum()), 'valid_boundary_pixels': int((edge & valid).sum()),
            'interior_valid_boundary_pixels': int((edge & interior).sum()),
            'changes': {name: {'added': int((current & ~initial).sum()),
                               'removed': int((initial & ~current).sum())}
                        for name, current, initial in [('building', building, initial_building),
                                                       ('valid', valid, initial_valid)]}}


def freeze_reviews(packet, edits, output, expected_packet, expected_manifest, expected_count=12):
    packet, output = Path(packet).resolve(), Path(output).resolve()
    private_root = Path(edits).resolve().parent
    require(output != private_root and output.is_relative_to(private_root), 'Keep submission snapshots beside the private review drafts.')
    require(not output.exists(), 'Use a fresh submission snapshot; existing reviews are preserved.')
    require(sha256(packet / 'packet.json') == expected_packet and sha256(packet / 'manifest.jsonl') == expected_manifest,
            'Prepared training cohort changed.')
    rows = read_jsonl(packet / 'manifest.jsonl')
    require_development(rows, {'train'})
    store = TrainingReviewStore(packet, edits)
    batch = store.batch_state()
    require(batch['ready'] == batch['total'] == len(rows) == expected_count, 'Finish all current review versions first.')
    require({row['sample_id'] for row in rows} == {selected.sid for selected in store.stores.values()},
            'Review images differ from the training manifest.')
    progress = store._progress()
    pins = {str(store.progress_path): sha256(store.progress_path),
            str(packet / 'packet.json'): expected_packet, str(packet / 'manifest.jsonl'): expected_manifest}
    submissions = []
    for view in batch['views']:
        number = view['number']
        selected = store.for_view(number)
        review = store.review_state(number)
        mark = progress['completed'][str(number)]
        version = Path(mark['output_directory']).resolve()
        require(version.is_relative_to(selected.output.resolve() / 'versions'), 'Version escaped its review directory.')
        scene_path = selected.output / 'scene-review.json'
        require(sha256(scene_path) == mark['context_sha256'] and review['revision'] == mark['context_revision'],
                'Completed scene-review identity changed.')
        history = selected.output / 'scene-review-history' / f"revision-{review['revision']:06d}.json"
        require(sha256(history) == sha256(scene_path), 'Scene-review history differs from the completed response.')
        require(sha256(version / 'packet.json') == mark['packet_sha256'], 'Saved mask version changed.')
        version_packet = load_json(version / 'packet.json')
        require(len(version_packet['samples']) == 1, 'Expected a single exact-image version.')
        sample = version_packet['samples'][0]
        require(sample['sample_id'] == selected.sid and sample['number'] == number and
                version_packet['source_packet_sha256'] == expected_packet, 'Mask version belongs to another source.')
        paths = {}
        for key in ('target_image', 'building_mask', 'valid_mask', 'building_card', 'valid_card'):
            path = (version / sample[key]).resolve()
            require(path.is_relative_to(version) and sha256(path) == sample[key + '_sha256'], 'Saved review artifact changed.')
            pins[str(path)] = sample[key + '_sha256']
            paths[key] = str(path)
        require(sample['target_image_sha256'] == selected.sample['target_image_sha256'], 'Mask target differs from prepared view.')
        with Image.open(paths['target_image']) as image:
            shape = (image.height, image.width)
        building = binary_mask(paths['building_mask'], shape)
        valid = binary_mask(paths['valid_mask'], shape)
        require(valid.any(), 'No scored pixels in the completed review.')
        require(np.array_equal(building, unpack_mask(mark['masks']['building'], shape)) and
                np.array_equal(valid, unpack_mask(mark['masks']['valid'], shape)), 'Saved masks differ from completion pixels.')
        initial_building = binary_mask(packet / selected.sample['building_mask'], shape)
        initial_valid = binary_mask(packet / selected.sample['valid_mask'], shape)
        measurements = mask_measurements(building, valid, initial_building, initial_valid)
        require(measurements['changes'] == mark['changes'], 'Mask changes disagree with saved editor provenance.')
        for path in (scene_path, history, version / 'packet.json'):
            pins[str(path)] = sha256(path)
        submissions.append({'number': number, 'sample_id': selected.sid, 'human_review': review,
                            'version': mark['version'], 'completed_utc': mark['completed_utc'],
                            'packet_sha256': mark['packet_sha256'], 'paths': paths,
                            'measurements': measurements, 'training_eligible': False,
                            'acceptance_status': 'content_and_fitting_eligibility_pending'})
    require(all(sha256(path) == digest for path, digest in pins.items()), 'Review changed during verification; retain the new edit and retry.')
    supported = [entry for entry in submissions if entry['human_review']['pairing'] == 'supported'
                 and entry['human_review']['support'] == 'mostly_inside' and entry['human_review']['mask_review'] == 'checked']
    aggregate = {'created_utc': utc_now(), 'status': 'submitted_versions_verified_content_followup_pending',
                 'submitted_reviews': len(submissions), 'exact_mask_pairs_verified': len(submissions),
                 'human_supported_inside_checked': len(supported),
                 'pairing_counts': dict(Counter(entry['human_review']['pairing'] for entry in submissions)),
                 'coverage_counts': dict(Counter(entry['human_review']['support'] for entry in submissions)),
                 'positive_boundary_pixels_with_valid_neighborhood': sum(entry['measurements']['interior_valid_boundary_pixels'] for entry in supported),
                 'supported_views_with_zero_valid_neighborhood_boundary': sum(entry['measurements']['interior_valid_boundary_pixels'] == 0 for entry in supported),
                 'source_packet_sha256': expected_packet, 'source_manifest_sha256': expected_manifest,
                 'training_eligible_views': 0, 'new_training_updates': 0, 'sealed_images_opened': False}
    save_json(output / 'submissions.json', {'submissions': submissions, 'source_file_sha256': pins,
                                          'human_submitted_masks_modified': False})
    aggregate['snapshot_sha256'] = sha256(output / 'submissions.json')
    save_json(output / 'summary.json', aggregate)
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    preparation = load_json('reports/TRAINING_REVIEW_STATUS.json')
    import json
    print(json.dumps(freeze_reviews('data/review/training12-v1', 'data/review/training12-edits', args.output,
                                  preparation['packet_sha256'], preparation['manifest_sha256']), indent=2))
