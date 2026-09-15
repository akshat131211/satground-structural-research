"""Prepare unannotated training-only pairs for a later building-identity study."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw

from satground.camera import crop_panorama
from satground.common import (ResearchError, object_hash, read_jsonl, require_development,
                              safe_data_path, save_json, sha256, utc_now, write_jsonl)


def select_tiles(rows, per_city=8):
    require_development(rows, {'train'})
    if per_city < 1:
        raise ResearchError('The number of review tiles per city must be positive.')
    tiles = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tiles[row['satellite']][row['panorama']].append(row)
    eligible = []
    for satellite, panoramas in sorted(tiles.items()):
        # The existing camera convention stores west as -90, equivalent to 270.
        # Normalize only for coverage/order checks, preserving the source rows.
        complete = {p: sorted(v, key=lambda r: r['yaw'] % 360) for p, v in panoramas.items()
                    if len(v) == 4 and {r['yaw'] % 360 for r in v} == {0, 90, 180, 270}}
        if len(complete) < 2:
            continue
        if len({r['geo_group'] for values in panoramas.values() for r in values}) != 1:
            raise ResearchError('An overhead tile crosses geographic groups.')
        def separation(pair):
            a, b = [complete[p][0] for p in pair]
            return (a['h_offset'] - b['h_offset'])**2 + (a['w_offset'] - b['w_offset'])**2
        pair = min(combinations(sorted(complete), 2), key=lambda p: (-separation(p), p))
        if separation(pair) <= 0:
            continue
        selected = complete[pair[0]] + complete[pair[1]]
        eligible.append(dict(satellite=satellite, rows=selected,
            separation_offset_pixels=separation(pair)**.5, city=selected[0]['city'], geo_group=selected[0]['geo_group']))
    chosen, used_groups, used_panoramas = [], set(), set()
    cities = sorted({r['city'] for r in rows})
    for city in cities:
        count = 0
        for tile in sorted((t for t in eligible if t['city'] == city), key=lambda t: (t['geo_group'], t['satellite'])):
            panos = {r['panorama'] for r in tile['rows']}
            if tile['geo_group'] in used_groups or panos & used_panoramas:
                continue
            chosen.append(tile)
            used_groups.add(tile['geo_group'])
            used_panoramas.update(panos)
            count += 1
            if count == per_city:
                break
        if count != per_city:
            raise ResearchError(f'Insufficient distinct training groups with two camera positions in {city}.')
    return chosen, dict(source_tiles=len(tiles),
        source_unique_panoramas=len({r['panorama'] for r in rows}),
        eligible_multiview_tiles=len(eligible),
        eligible_by_city=dict(Counter(t['city'] for t in eligible)))


def prepare(manifest, data_root, output, per_city=8):
    rows = read_jsonl(manifest)
    chosen, inventory = select_tiles(rows, per_city)
    out = Path(output)
    if out.exists():
        raise ResearchError('Review packet exists; do not overwrite annotations.')
    chosen_rows = [row for tile in chosen for row in tile['rows']]
    for row in chosen_rows:
        for key in ('satellite', 'panorama'):
            if not safe_data_path(data_root, row[key]).is_file():
                raise ResearchError('Missing training review image.')
    for name in ('targets', 'overhead', 'sheets', 'annotations'):
        (out / name).mkdir(parents=True)
    write_jsonl(out / 'manifest.jsonl', chosen_rows)
    index, source_hashes = [], {}
    for number, tile in enumerate(chosen, 1):
        tile_id = object_hash(tile['satellite'])[:16]
        satellite = safe_data_path(data_root, tile['satellite'])
        with Image.open(satellite) as image:
            overhead = image.convert('RGB')
        overhead.save(out / 'overhead' / f'{tile_id}.png')
        source_hashes[tile['satellite']] = sha256(satellite)
        canvas = Image.new('RGB', (5 * 220, 2 * 252 + 60), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 6), f"Training tile {number:02d}: {tile['city']} | two real panoramas, four directions each", fill='black')
        draw.text((5, 23), f"Camera offset separation: {tile['separation_offset_pixels']:.1f} source pixels (not verified metres)", fill='black')
        views = []
        for j, row in enumerate(tile['rows']):
            row_number, col = divmod(j, 4)
            if col == 0:
                canvas.paste(overhead.resize((220, 220)), (0, 60 + row_number * 252))
            pano_path = safe_data_path(data_root, row['panorama'])
            source_hashes[row['panorama']] = sha256(pano_path)
            with Image.open(pano_path) as pano:
                target = Image.fromarray(crop_panorama(pano.convert('RGB'), row['yaw'], row['pitch'], row['fov'], row['size']))
            path = out / 'targets' / f"{row['sample_id']}.png"
            target.save(path)
            canvas.paste(target.resize((220, 220)), ((col + 1) * 220, 60 + row_number * 252))
            draw.text(((col + 1) * 220 + 3, 60 + row_number * 252 + 224),
                      f"Camera {row_number + 1}, yaw {row['yaw']:g}", fill='black')
            views.append(dict(sample_id=row['sample_id'], camera_number=row_number + 1,
                              target_sha256=sha256(path), reviewed=False, notes=''))
        canvas.save(out / 'sheets' / f'tile-{number:02d}.jpg', quality=92)
        index.append(dict(tile_id=tile_id, sheet=f'sheets/tile-{number:02d}.jpg',
            overhead_sha256=sha256(out / 'overhead' / f'{tile_id}.png'),
            source_satellite=tile['satellite'], geo_group=tile['geo_group'], views=views,
            reviewed=False, reviewer='', reviewed_utc='', visible_roof_regions=[],
            cross_view_correspondences=[], unknown_regions=[],
            note='Unannotated. Observed roof regions are not verified ground footprints; no instance IDs or masks are supplied.'))
    save_json(out / 'identity-review.pending.json', index)
    save_json(out / 'source-image-hashes.json', source_hashes)
    summary = dict(created_utc=utc_now(), **inventory, selected_tiles=len(chosen),
        selected_panoramas=len({r['panorama'] for r in chosen_rows}), selected_views=len(chosen_rows),
        selected_geographic_groups=len({t['geo_group'] for t in chosen}),
        selected_tiles_by_city=dict(Counter(t['city'] for t in chosen)),
        source_manifest_sha256=sha256(manifest), selected_manifest_sha256=sha256(out / 'manifest.jsonl'),
        source_images_digest=object_hash(source_hashes),
        selection='Eight tiles per city in distinct training groups, sorted by group/tile; farthest metadata-offset panorama pair, all four existing orientations. No image, mask, model, or metric filtering.',
        camera_offsets_have_verified_metric_scale=False,
        reviewed_count=0, masks_created=False, correspondences_created=False,
        training_eligible=False, validation_calibration_audit_or_Seattle_opened=False)
    save_json(out / 'packet.json', summary)
    (out / 'START_HERE.md').write_text(
        '# Training-only multiple-view review\n\n'
        'This packet contains two distinct real panorama positions per overhead tile and all four existing perspective directions. It is a candidate review cohort, not a newly validated training set.\n\n'
        'Open sheets/ to compare cameras, then inspect the original-resolution overhead/ and targets/. Record whether buildings can be linked unambiguously across views and to an observed roof region. Roof pixels must not be described as measured footprints or heights. Mark hidden, outside-crop, misaligned, changed, or uncertain structures unknown.\n\n'
        'identity-review.pending.json contains empty records. It has no building masks, assigned instance IDs, or verified correspondences. Edit a copy only after actual review and retain the source hashes. A future validated annotation contract and trainer are still required before this packet can provide identity supervision.\n\n'
        'This supplements the original data-quality review; it does not complete footprint, pairing, duplicate, or manual evaluation requirements. Keep images and annotations local.\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/pilot/train.jsonl')
    parser.add_argument('--data-root', default='data/vigor')
    parser.add_argument('--output', default='data/review/identity24')
    args = parser.parse_args()
    print(prepare(args.manifest, args.data_root, args.output))
