"""Metadata-only geographic splits, followed by explicit local RGB verification."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (ResearchError, TRAIN_CITIES, object_hash, read_jsonl,
                     require_development, safe_data_path, save_json, sha256, write_jsonl)


def parse_pairs(path):
    result = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        values = line.split()
        if len(values) != 5:
            raise ResearchError(f"Line {index + 1}: expected explicit camera offsets, got {len(values)} fields.")
        sat, pano, sky, h, w = values
        city = sat.split('/')[0]
        if city not in TRAIN_CITIES:
            raise ResearchError(f"Non-development city in training metadata: {city}")
        try:
            lat, lon = map(float, Path(sat).stem.split('_')[-2:])
            h, w = float(h), float(w)
        except ValueError as exc:
            raise ResearchError(f"Malformed location at line {index + 1}") from exc
        if abs(h) > 320 or abs(w) > 320 or not (-90 < lat < 90 and -180 < lon < 180):
            raise ResearchError(f"Invalid location at line {index + 1}")
        result.append(dict(city=city, satellite=sat, panorama=pano, sky_mask=sky,
                           latitude=lat, longitude=lon, h_offset=h, w_offset=w))
    return result


def prepare(metadata, output, counts=None, seed=17, block_m=1000, guard_m=100, size=128, views=4):
    """Discard every tile close to a block edge, even within the same partition.

    Guard is a conservative *assumption*, recorded for subsequent footprint QA.
    Connected panorama IDs spanning blocks are excluded rather than split.
    """
    if guard_m <= 0 or block_m <= 2 * guard_m or views not in (1, 2, 4):
        raise ResearchError("Invalid spatial block, guard distance, or view count.")
    counts = counts or dict(train=2000, validation=250, calibration=250, audit=500)
    raw = parse_pairs(metadata)
    groups, pano_groups, rejected = defaultdict(list), defaultdict(set), Counter()
    # Fixed local map per city; enough for conservative kilometre blocks, not survey geometry.
    centers = {city: (np.mean([r['latitude'] for r in raw if r['city'] == city]),
                      np.mean([r['longitude'] for r in raw if r['city'] == city])) for city in TRAIN_CITIES if any(r['city'] == city for r in raw)}
    for row in raw:
        lat0, lon0 = centers[row['city']]
        x = (row['longitude'] - lon0) * 111320 * math.cos(math.radians(lat0))
        y = (row['latitude'] - lat0) * 111320
        group = f"{row['city']}:{math.floor(x / block_m)}:{math.floor(y / block_m)}"
        row['geo_group'] = group
        pano_groups[row['panorama']].add(group)
        if min(x % block_m, block_m - x % block_m, y % block_m, block_m - y % block_m) < guard_m:
            rejected['edge_guard_pairs'] += 1
            continue
        groups[group].append(row)
    # Exclude a panorama that also exists in a discarded neighbouring block.
    for group in list(groups):
        groups[group] = [r for r in groups[group] if len(pano_groups[r['panorama']]) == 1]
        if not groups[group]:
            del groups[group]
    partitions = defaultdict(list)
    for city in sorted(centers):
        keys = sorted([g for g in groups if g.startswith(city + ':')], key=lambda g: object_hash([seed, g]))
        # Assign whole blocks; reserve a useful fraction of each city for each holdout.
        for index, group in enumerate(keys):
            fraction = (index + .5) / len(keys)
            split = 'train' if fraction < .60 else 'validation' if fraction < .73 else 'calibration' if fraction < .86 else 'audit'
            partitions[split].extend(groups[group])
    all_rows, selected_counts = [], {}
    for split, requested in counts.items():
        candidates = partitions[split]
        tiles = sorted({r['satellite'] for r in candidates}, key=lambda tile: object_hash([seed, split, tile]))
        selected = set(tiles[:requested])
        selected_counts[split] = dict(requested_tiles=requested, available_tiles=len(tiles), selected_tiles=len(selected))
        unique_pairs = {}
        for row in candidates:
            if row['satellite'] in selected:
                unique_pairs[(row['satellite'], row['panorama'])] = row
        for row in unique_pairs.values():
            for yaw in [0, 90, 180, -90][:views]:
                sample = dict(row, split=split, yaw=float(yaw), pitch=0.0, fov=90.0, size=size,
                              sample_id=object_hash([row['satellite'], row['panorama'], yaw, size])[:24],
                              label_provenance='not_generated', source='VIGOR_original_RGB_required',
                              camera_convention='pinned_Sat3DGen_VIGOR', orientation='release_north_alignment_assumed',
                              footprint_radius_bound_m=guard_m, footprint_bound_verified=False)
                all_rows.append(sample)
    all_rows.sort(key=lambda r: (r['split'], r['sample_id']))
    require_development(all_rows)
    assert_disjoint(all_rows)
    out = Path(output)
    for split in counts:
        write_jsonl(out / f'{split}.jsonl', [r for r in all_rows if r['split'] == split])
    write_jsonl(out / 'all.jsonl', all_rows)
    report = dict(metadata_sha256=sha256(metadata), seed=seed, block_m=block_m, guard_m=guard_m,
                  counts=selected_counts, pairs_in_source=len(raw), rows_written=len(all_rows),
                  excluded=dict(rejected), rgb_verified=False, footprint_bound_verified=False,
                  Seattle_opened=False, holdout_may_have_been_seen_in_pretraining=True)
    save_json(out / 'split-report.json', report)
    return report


def assert_disjoint(rows):
    for key in ('satellite', 'panorama', 'geo_group'):
        seen = {}
        for row in rows:
            value = row[key]
            if value in seen and seen[value] != row['split']:
                raise ResearchError(f"Split leakage through {key}: {value}")
            seen[value] = row['split']


def validate_rgb(manifest, data_root, output):
    rows = read_jsonl(manifest)
    require_development(rows)
    assert_disjoint(rows)
    paths = sorted({r[k] for r in rows for k in ('satellite', 'panorama')})
    split_for = {r[k]: r['split'] for r in rows for k in ('satellite', 'panorama')}
    missing, bad, records, digest_splits, near = [], [], [], defaultdict(set), []
    fingerprints = defaultdict(list)
    for relative in paths:
        path = safe_data_path(data_root, relative)
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
                if min(width, height) < 128:
                    raise ValueError('Image too small')
                # dHash is a screening signal; collisions require review, never a similarity label.
                grey = np.asarray(img.convert('L').resize((9, 8)), dtype=np.int16)
                phash = int.from_bytes(np.packbits(grey[:, 1:] > grey[:, :-1]).tobytes(), 'big')
            digest = sha256(path)
            digest_splits[digest].add(split_for[relative])
            records.append(dict(path=relative, sha256=digest, width=width, height=height))
            kind = 'satellite' if '/satellite/' in relative else 'panorama'
            fingerprints[kind].append((relative, split_for[relative], phash))
        except Exception as exc:
            bad.append(dict(path=relative, error=str(exc)))
    # Hash locality buckets find Hamming-distance <= 3 across splits (4 x 16-bit pigeonhole).
    for items in fingerprints.values():
        buckets = defaultdict(list)
        compared = set()
        for index, (path, split, fingerprint) in enumerate(items):
            for part in range(4):
                key = (part, (fingerprint >> (16 * part)) & 65535)
                for previous in buckets[key]:
                    if (previous, index) in compared:
                        continue
                    compared.add((previous, index))
                    ppath, psplit, other = items[previous]
                    if split != psplit and (fingerprint ^ other).bit_count() <= 3:
                        near.append([ppath, path])
                buckets[key].append(index)
    duplicates = [h for h, splits in digest_splits.items() if len(splits) > 1]
    report = dict(manifest_sha256=sha256(manifest), data_root=str(Path(data_root).resolve()),
                  missing_count=len(missing), missing_examples=missing[:20], invalid=bad,
                  cross_split_duplicate_hashes=duplicates, near_duplicate_review_candidates=near,
                  files=records, status='ready_for_review' if not missing and not bad and not duplicates else 'blocked',
                  near_duplicates_human_reviewed=False, footprint_bound_verified=False)
    save_json(output, report)
    return report


def require_files(rows, data_root):
    missing = [str(safe_data_path(data_root, r[k])) for r in rows for k in ('satellite', 'panorama')
               if not safe_data_path(data_root, r[k]).is_file()]
    if missing:
        raise ResearchError(f"Original VIGOR RGB images are required; {len(set(missing))} files missing. First: {missing[0]}")
