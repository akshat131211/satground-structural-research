"""Create local development images and unfilled review records; never certify annotations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from satground.camera import crop_panorama
from satground.common import (ResearchError, read_jsonl, require_development,
                             safe_data_path, save_json, sha256, utc_now, write_jsonl)


def select_views(rows, per_split=100):
    require_development(rows, {'train', 'validation'})
    chosen, panoramas, tiles = [], set(), set()
    for split in ('train', 'validation'):
        groups = defaultdict(deque)
        for row in sorted(rows, key=lambda r: r['sample_id']):
            if row['split'] == split:
                groups[(row['city'], row['geo_group'])].append(row)
        by_city = defaultdict(list)
        for key in sorted(groups):
            by_city[key[0]].append(key)
        cities = sorted(by_city)
        order = [by_city[city][i] for i in range(max((len(v) for v in by_city.values()), default=0))
                 for city in cities if i < len(by_city[city])]
        selected = 0
        while selected < per_split:
            progress = False
            for key in order:
                queue = groups[key]
                while queue and (queue[0]['panorama'] in panoramas or queue[0]['satellite'] in tiles):
                    queue.popleft()
                if queue:
                    row = queue.popleft()
                    chosen.append(row)
                    panoramas.add(row['panorama'])
                    tiles.add(row['satellite'])
                    selected += 1
                    progress = True
                    if selected == per_split:
                        break
            if not progress:
                raise ResearchError('Insufficient unique development panoramas and tiles for review.')
    return chosen


def prepare(manifest_dir, data_root, output):
    source = Path(manifest_dir)
    rows = read_jsonl(source / 'train.jsonl') + read_jsonl(source / 'validation.jsonl')
    rows = select_views(rows)
    out = Path(output).resolve()
    if out.exists():
        raise ResearchError('Review directory exists; preserve all annotations and use a new output directory.')
    for row in rows:
        for key in ('satellite', 'panorama'):
            if not safe_data_path(data_root, row[key]).is_file():
                raise ResearchError('Missing review image: ' + row[key])
    for folder in ('targets', 'satellite', 'masks', 'sheets'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'manifest.jsonl', rows)
    index, checklist = {}, []
    for number, row in enumerate(rows):
        sid = row['sample_id']
        with Image.open(safe_data_path(data_root, row['panorama'])) as pano:
            target = Image.fromarray(crop_panorama(pano.convert('RGB'), row['yaw'], row['pitch'], row['fov'], row['size']))
        target_path = out / 'targets' / f'{sid}.png'
        target.save(target_path)
        with Image.open(safe_data_path(data_root, row['satellite'])) as satellite:
            satellite.convert('RGB').save(out / 'satellite' / f'{sid}.png')
        index[sid] = dict(reviewed=False, reviewer='', reviewed_utc='',
            building_mask=str(out / 'masks' / f'{sid}-building.png'),
            valid_mask=str(out / 'masks' / f'{sid}-valid.png'),
            target_image=str(target_path), target_image_sha256=sha256(target_path),
            building_mask_sha256='', valid_mask_sha256='', annotation_method='manual_drawing_pending',
            target_sha256=sha256(target_path), instance_errors=None,
            notes='Unannotated. Create exact-size binary masks and review them before setting reviewed=true.')
        checklist.append(dict(number=number + 1, sample_id=sid, split=row['split'], city=row['city'],
            geo_group=row['geo_group'], yaw=row['yaw'], pitch=row['pitch'], fov=row['fov'],
            h_offset=row['h_offset'], w_offset=row['w_offset'],
            pairing_status='', orientation_status='', temporal_mismatch='',
            visible_structure_notes='', reviewed_by='', reviewed_utc=''))
    save_json(out / 'manual-index.pending.json', index)
    with (out / 'pairing-checklist.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checklist[0]))
        writer.writeheader()
        writer.writerows(checklist)
    for offset in range(0, len(rows), 10):
        canvas = Image.new('RGB', (1024, 5 * 290), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(rows[offset:offset + 10]):
            x, y = (j % 2) * 512, (j // 2) * 290
            sid = row['sample_id']
            for col, directory in enumerate(('satellite', 'targets')):
                with Image.open(out / directory / f'{sid}.png') as im:
                    canvas.paste(im.resize((256, 256)), (x + col * 256, y + 30))
            draw.text((x + 4, y + 2), f'{offset + j + 1:03d} {row["city"]} {row["split"]} yaw={row["yaw"]}', fill='black')
            draw.text((x + 4, y + 15), f'{sid} | satellite / real ground', fill='black')
        canvas.save(out / 'sheets' / f'page-{offset // 10 + 1:02d}.jpg', quality=92)
    summary = dict(created_utc=utc_now(), views=len(rows),
        selection='Round-robin cities/geographic groups; sample-ID order within groups; unique tiles and panoramas; no metric or content filtering.',
        split_counts=dict(Counter(r['split'] for r in rows)), city_counts=dict(Counter(r['city'] for r in rows)),
        geographic_groups=len({r['geo_group'] for r in rows}), reviewed_count=0,
        human_review_complete=False, manual_masks_created=False,
        source_manifests={split: sha256(source / f'{split}.jsonl') for split in ('train', 'validation')},
        review_manifest_sha256=sha256(out / 'manifest.jsonl'), audit_or_Seattle_opened=False)
    save_json(out / 'packet.json', summary)
    (out / 'START_HERE.md').write_text(
        '# Development review packet\n\n'
        '200 real development views: 100 training and 100 validation. No masks have been annotated or verified.\n\n'
        '1. Open the numbered sheets for a quick overview, then inspect the full images in targets/ and satellite/.\n'
        '2. Record pairing/orientation errors and suspected date differences in pairing-checklist.csv. Use unknown where evidence is absent. Do not exclude a difficult case just because a model performs poorly.\n'
        '3. At the exact target resolution, draw masks/<sample_id>-building.png with visible building pixels white and other pixels black. Draw the matching -valid.png with assessable static pixels white; uncertain, occluded, and dynamic pixels black. An image without visible buildings needs an all-black building mask and an honestly reviewed valid mask.\n'
        '4. Work on a copy of manual-index.pending.json. Mark only completed entries reviewed=true, with your name and actual UTC date. For partial evaluation, include only completed entries in the index. Never submit placeholder masks.\n'
        '5. Keep the packet local. Do not upload VIGOR images or derived masks to GitHub.\n\n'
        'The index supports evaluation of this exact review manifest. Main data QA also requires separate footprint and duplicate review; see docs/REVIEW_GUIDE.md.\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', default='data/manifests/pilot')
    parser.add_argument('--data-root', default='data/vigor')
    parser.add_argument('--output', default='data/review/development200')
    args = parser.parse_args()
    print(prepare(args.manifest_dir, args.data_root, args.output))
