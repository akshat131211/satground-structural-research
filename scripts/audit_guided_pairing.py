"""Audit source/camera consistency of the three guided pairing observations.

Agreement with the release is a software check, not independent validation of
heading, footprint scale, building identity, or capture date.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from satground.camera import camera_matrices, crop_panorama
from satground.common import (ResearchError, check_vendor, load_json, read_jsonl,
                              require_development, safe_data_path, save_json, sha256)
from satground.manifests import parse_pairs


def audit(output='data/review/guided-first3/camera-audit.json'):
    sys.path.insert(0, str(check_vendor()))
    from source.rendering.pano2perspective import GetPerspective
    from source.rendering.transform_perspective import compose_rotmat

    review = load_json('data/review/guided-first3/review.json')
    manifest = read_jsonl('data/review/identity24/manifest.jsonl')
    require_development(manifest, {'train'})
    by_id = {r['sample_id']: r for r in manifest}
    metadata = Path('data/metadata/train__corrected_all_3city_remove_building.txt')
    split_record = load_json('data/manifests/pilot/split-report.json')
    if sha256(metadata) != split_record['metadata_sha256']:
        raise ResearchError('Source pairing metadata changed.')
    pairs = {(r['satellite'], r['panorama']): r for r in parse_pairs(metadata)}
    results = []
    for check in review['checks']:
        rows = [by_id[v['sample_id']] for v in check['views']]
        if len(rows) != 2 or rows[0]['satellite'] != rows[1]['satellite']:
            raise ResearchError('Expected two real cameras associated with one overhead tile.')
        cameras = []
        for entry, row in zip(check['views'], rows):
            source = pairs[(row['satellite'], row['panorama'])]
            if any(row[key] != value for key, value in source.items()):
                raise ResearchError('Guided camera differs from the released pairing record.')
            path = safe_data_path('data/vigor', row['panorama'])
            with Image.open(path) as image:
                pano = np.asarray(image.convert('RGB'))
            ours = crop_panorama(pano, row['yaw'], row['pitch'], row['fov'], row['size'])
            target = Path('data/review/identity24/targets') / f"{row['sample_id']}.png"
            if sha256(target) != entry['target_sha256'] or not np.array_equal(ours, np.asarray(Image.open(target))):
                raise ResearchError('Reviewed target differs from its exact source crop.')
            matrix, k, _ = camera_matrices(row['h_offset'], row['w_offset'], row['yaw'], row['pitch'], row['fov'], row['size'])
            reference = GetPerspective(pano, [k[0, 0], k[1, 1], k[0, 2], k[1, 2]], row['yaw'], row['pitch'], row['size'], row['size'])
            difference = np.abs(ours.astype(float) - reference.astype(float))
            rotation_error = float(np.abs(matrix[:3, :3] - compose_rotmat(0, row['pitch'], row['yaw'])).max())
            cameras.append(dict(sample_id=row['sample_id'], source_pair_verified=True,
                crop_uint8_max_difference_from_release=float(difference.max()),
                crop_mean_absolute_difference_from_release=float(difference.mean()),
                rotation_max_difference_from_release=rotation_error,
                h_offset=row['h_offset'], w_offset=row['w_offset'], yaw=row['yaw'], pitch=row['pitch'], fov=row['fov'],
                original_panorama_sha256=sha256(path), target_sha256=sha256(target)))
        a, b = rows
        offset_distance = math.hypot(a['h_offset'] - b['h_offset'], a['w_offset'] - b['w_offset'])
        results.append(dict(check_number=check['check_number'], cameras=cameras,
            distinct_panorama_ids=a['panorama'] != b['panorama'], distinct_camera_positions=offset_distance > 0,
            camera_separation_source_pixels=offset_distance, same_requested_yaw=a['yaw'] == b['yaw'],
            human_observation=check['human_answer'], identity_confirmed=False,
            conclusion='Source rows and crop implementation checked; different camera positions can see different facades. Independent real heading and building correspondence remain unresolved.'))
    record = dict(checks=results, source_metadata_sha256=sha256(metadata),
        independent_heading_verified=False, metric_scale_verified=False, capture_dates_verified=False,
        samples_removed=0, human_review_complete=False, implementation_check_only=True)
    save_json(output, record)
    print({r['check_number']: dict(separation_source_pixels=r['camera_separation_source_pixels'],
        max_crop_difference=max(c['crop_uint8_max_difference_from_release'] for c in r['cameras']),
        mean_crop_difference=max(c['crop_mean_absolute_difference_from_release'] for c in r['cameras'])) for r in results})


if __name__ == '__main__':
    audit()
