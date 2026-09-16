"""Check reviewed target/camera preprocessing against the pinned release.

This does not certify real-world heading, capture dates, scale, or pairing.
Exact identifiers and camera fields stay in the local output; the aggregate
contains only counts, errors, and evidence hashes.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from satground.annotations import verified_reviewed_masks
from satground.camera import camera_matrices, crop_panorama
from satground.common import (ResearchError, check_vendor, load_json, read_jsonl,
    require_development, safe_data_path, save_json, sha256, utc_now)
from satground.manifests import parse_pairs


def audit(index_path, manifest_path, output, aggregate, data_root='data/vigor'):
    if Path(output).exists() or Path(aggregate).exists():
        raise ResearchError('Choose fresh audit output files.')
    sys.path.insert(0, str(check_vendor()))
    from source.rendering.pano2perspective import GetPerspective
    from source.rendering.transform_perspective import compose_rotmat

    rows = read_jsonl(manifest_path)
    require_development(rows, {'train', 'validation'})
    by_id = {r['sample_id']: r for r in rows}
    index = load_json(index_path)
    if not index or not set(index).issubset(by_id):
        raise ResearchError('Reviewed index is empty or outside the development manifest.')
    metadata = Path('data/metadata/train__corrected_all_3city_remove_building.txt')
    if sha256(metadata) != load_json('data/manifests/pilot/split-report.json')['metadata_sha256']:
        raise ResearchError('Source pairing metadata changed.')
    pairs = {(r['satellite'], r['panorama']): r for r in parse_pairs(metadata)}
    checked = []
    for sid, entry in index.items():
        row = by_id[sid]
        source = pairs.get((row['satellite'], row['panorama']))
        if source is None or any(row.get(k) != v for k, v in source.items()):
            raise ResearchError('A reviewed view differs from the released source record.')
        path = safe_data_path(data_root, row['panorama'])
        with Image.open(path) as im:
            pano = np.asarray(im.convert('RGB'))
        crop = crop_panorama(pano, row['yaw'], row['pitch'], row['fov'], row['size'])
        verified_reviewed_masks(entry, crop)
        matrix, k, _ = camera_matrices(row['h_offset'], row['w_offset'], row['yaw'],
                                      row['pitch'], row['fov'], row['size'])
        reference = GetPerspective(pano, [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                                   row['yaw'], row['pitch'], row['size'], row['size'])
        crop_error = float(np.abs(crop.astype(float) - reference.astype(float)).max())
        rotation_error = float(np.abs(matrix[:3, :3] - compose_rotmat(0, row['pitch'], row['yaw'])).max())
        if crop_error != 0 or rotation_error > 1e-6:
            raise ResearchError('Reviewed-view preprocessing differs from the release.')
        checked.append(dict(sample_id=sid, review_view_number=entry.get('review_view_number'),
            source_pair_record_matches=True, original_panorama_sha256=sha256(path),
            target_sha256=entry['target_image_sha256'], crop_max_difference=crop_error,
            rotation_max_difference=rotation_error,
            camera={k: row[k] for k in ('h_offset', 'w_offset', 'yaw', 'pitch', 'fov', 'size')}))
    raw = dict(created_utc=utc_now(), views=checked, manual_index_sha256=sha256(index_path),
        manifest_sha256=sha256(manifest_path), metadata_sha256=sha256(metadata),
        implementation_consistency_only=True, independent_heading_verified=False,
        real_world_pairing_verified=False, metric_scale_verified=False,
        capture_dates_verified=False, full_data_qa_complete=False)
    save_json(output, raw)
    public = {k: v for k, v in raw.items() if k != 'views'}
    public.update(views_checked=len(checked), source_pair_records_matched=len(checked),
        maximum_crop_difference=max(r['crop_max_difference'] for r in checked),
        maximum_rotation_difference=max(r['rotation_max_difference'] for r in checked),
        local_audit_sha256=sha256(output), audit_script_sha256=sha256(__file__))
    save_json(aggregate, public)
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'manifest', 'output', 'aggregate'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--data-root', default='data/vigor')
    args = parser.parse_args()
    result = audit(args.index, args.manifest, args.output, args.aggregate, args.data_root)
    print('Source records and camera crops match the release for', result['views_checked'], 'reviewed views.')
