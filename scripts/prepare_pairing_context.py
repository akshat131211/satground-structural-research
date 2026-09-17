"""Draw actual-target-yaw context for the expanded reviewed development cohort.

These diagrams are correspondence aids, not georeferenced building annotations.
All source images and exact camera records are written to ignored local data.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from satground.annotations import verified_reviewed_masks
from satground.camera import camera_matrices, crop_panorama
from satground.common import (ResearchError, check_vendor, load_json, read_jsonl,
                             require_development, safe_data_path, save_json, sha256, utc_now)


def ray_exit(origin, direction, bounds=(0., 640.)):
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    if not np.isfinite(origin).all() or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
        raise ResearchError('A finite nonzero image-plane ray is required.')
    low, high = bounds
    if not (low <= origin).all() or not (origin <= high).all():
        raise ResearchError('Camera lies outside the requested diagram bounds.')
    distances = [((high if d > 0 else low) - p) / d
                 for p, d in zip(origin, direction) if abs(d) > 1e-10]
    t = min(distances)
    return origin + t * direction


def geometry(row, yaw):
    c2w, _, _ = camera_matrices(row['h_offset'], row['w_offset'], yaw, 0, 90, 256)
    origin = np.array([320 + row['w_offset'], 320 + row['h_offset']], float)
    directions = []
    for x in (-1., 0., 1.):
        xyz = c2w[:3, :3] @ np.array([x, 0, 1.])
        xy = xyz[:2] * [1., -1.]  # Pinned sampler flips world Y into image-row coordinates.
        directions.append(xy / np.linalg.norm(xy))
    exits = [ray_exit(origin, d) for d in directions]
    return origin, directions, exits


def prepare(index_path, manifest_path, output):
    check_vendor()
    out = Path(output)
    if out.exists():
        raise ResearchError('Choose a new context packet directory.')
    rows = read_jsonl(manifest_path)
    require_development(rows, {'train', 'validation'})
    by_id = {r['sample_id']: r for r in rows}
    index = load_json(index_path)
    if not index or not set(index).issubset(by_id):
        raise ResearchError('Review index must belong to this development manifest.')
    out.mkdir(parents=True)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    entries = []
    colors = ('#ffd859', '#ff7878', '#74dfa9', '#70b7ff')
    for sid, entry in sorted(index.items(), key=lambda pair: pair[1]['review_view_number']):
        row, number = by_id[sid], entry['review_view_number']
        # +180 and -180 are geometrically equal but can differ after uint8
        # interpolation. Preserve the exact recorded yaw for the reviewed crop.
        yaws = (row['yaw'], *(((row['yaw'] + turn + 180) % 360) - 180 for turn in (90, 180, -90)))
        if (row['pitch'], row['fov'], row['size']) != (0, 90, 256):
            raise ResearchError('This context layout expects zero-pitch, 90-degree, 256-pixel crops.')
        sat_path, pano_path = (safe_data_path('data/vigor', row[k]) for k in ('satellite', 'panorama'))
        with Image.open(sat_path) as im:
            satellite = im.convert('RGB')
        with Image.open(pano_path) as im:
            panorama = im.convert('RGB')
        if satellite.size != (640, 640):
            raise ResearchError('Release-coordinate diagram requires a 640-pixel tile.')
        target = crop_panorama(panorama, row['yaw'], 0, 90, 256)
        verified_reviewed_masks(entry, target)
        page = Image.new('RGB', (1280, 770), '#121922')
        draw = ImageDraw.Draw(page)
        draw.text((14, 10), f'View {number} | Satellite crop and real panorama directions | No generated images', font=font, fill='white')
        overlay = satellite.convert('RGBA')
        tint = Image.new('RGBA', satellite.size)
        mark = ImageDraw.Draw(tint)
        origin, directions, exits = geometry(row, row['yaw'])
        # Extend beyond the tile and clip; joining edge intersections can omit a corner.
        mark.polygon([tuple(origin), tuple(origin + directions[0]*2000),
                      tuple(origin + directions[2]*2000)], fill=(255, 216, 89, 40))
        for edge in (exits[0], exits[2]):
            mark.line([tuple(origin), tuple(edge)], fill=(255, 216, 89, 255), width=3)
        overlay = Image.alpha_composite(overlay, tint).convert('RGB')
        mark = ImageDraw.Draw(overlay)
        ray_records = []
        for j, (yaw, color) in enumerate(zip(yaws, colors)):
            origin, dirs, exits = geometry(row, yaw)
            end = origin + dirs[1] * 115
            mark.line([tuple(origin), tuple(end)], fill=color, width=4)
            normal = np.array([-dirs[1][1], dirs[1][0]])
            mark.polygon([tuple(end), tuple(end - dirs[1]*15 + normal*6), tuple(end - dirs[1]*15 - normal*6)], fill=color)
            mark.text(tuple(end + normal*8), str(yaw), font=font, fill=color, stroke_width=1, stroke_fill='black')
            crop = Image.fromarray(crop_panorama(panorama, yaw, 0, 90, 256))
            if j == 0 and not np.array_equal(np.asarray(crop), target):
                raise ResearchError('Context target differs from the accepted target pixels.')
            crop_path = out / f'view{number}-yaw{yaw}.png'
            crop.save(crop_path)
            x, y = 650 + (j % 2)*310, 48 + (j // 2)*320
            draw.text((x, y), f'Yaw {yaw}' + (' | reviewed target' if j == 0 else ''), font=font, fill=color)
            page.paste(crop.resize((292, 292), Image.Resampling.NEAREST), (x, y+23))
            padded_exit = ray_exit(origin, dirs[1], (-80., 720.))
            ray_records.append(dict(yaw=yaw, center_ray_direction=dirs[1].tolist(),
                center_ray_input_exit=exits[1].tolist(), distance_to_input_edge_px=float(np.linalg.norm(exits[1]-origin)),
                distance_to_padded_edge_px=float(np.linalg.norm(padded_exit-origin)), crop_sha256=sha256(crop_path)))
        x, y = origin
        mark.ellipse((x-7,y-7,x+7,y+7), fill='#ffffff', outline='black', width=2)
        page.paste(overlay, (0, 48))
        draw.text((12, 702), 'White dot: released camera. Yellow wedge: actual target yaw, 90-degree horizontal field of view.', font=font, fill='white')
        draw.text((12, 727), 'Arrows follow model coordinates. Absolute heading, distances in metres and building identities are unverified.', font=font, fill='#bbc7d5')
        page_path = out / f'view{number}-context.png'
        page.save(page_path)
        entries.append(dict(sample_id=sid, review_view_number=number, source_satellite_sha256=sha256(sat_path),
                            source_panorama_sha256=sha256(pano_path), camera_xy=origin.tolist(), target_yaw=row['yaw'], rays=ray_records,
                            context_image_sha256=sha256(page_path)))
    record = dict(created_utc=utc_now(), manifest_sha256=sha256(manifest_path), manual_index_sha256=sha256(index_path),
                  script_sha256=sha256(__file__), views=entries, north_alignment_assumed=True,
                  real_world_pairing_verified=False, metric_scale_verified=False,
                  model_padding_per_side_fraction=.125,
                  padding_note='Pinned checkpoint pad=0.125 gives learned peripheral context; it adds no observed RGB.',
                  annotation_status='assistant_context_diagrams_not_human_correspondence_labels')
    save_json(out / 'context.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', default='data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json')
    parser.add_argument('--manifest', default='data/manifests/learning100/validation.jsonl')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print('Prepared', len(prepare(args.index, args.manifest, args.output)['views']), 'context diagrams.')
