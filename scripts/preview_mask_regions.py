"""Show exact target pixels, proposed labels, and changes in named review regions.

This is a read-only annotation aid. It never edits masks or records acceptance.
Nearest-neighbor enlargement exposes the actual scoring pixels, not new detail.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from satground.annotations import binary_mask
from satground.common import ResearchError, load_json, safe_data_path, save_json, sha256


def sample_images(root, sid):
    packet = load_json(root / 'packet.json')
    matches = [s for s in packet['samples'] if s['sample_id'] == sid]
    if len(matches) != 1:
        raise ResearchError('Expected exactly one sample in each packet.')
    sample = matches[0]
    paths = {}
    for key in ('target_image', 'building_mask', 'valid_mask'):
        paths[key] = safe_data_path(root, sample[key])
        if sha256(paths[key]) != sample[key + '_sha256']:
            raise ResearchError('Review pixels differ from packet checksums.')
    with Image.open(paths['target_image']) as image:
        target = np.asarray(image.convert('RGB'))
    return target, {c: binary_mask(paths[c + '_mask'], target.shape[:2])
                    for c in ('building', 'valid')}


def preview(packet, previous, sid, regions_path, output):
    root, parent, out = Path(packet), Path(previous), Path(output)
    if out.exists() or out.with_suffix('.json').exists():
        raise ResearchError('Use a fresh preview path to preserve review evidence.')
    target, masks = sample_images(root, sid)
    old_target, old_masks = sample_images(parent, sid)
    if not np.array_equal(target, old_target):
        raise ResearchError('Review revisions must show the same exact target.')
    regions = load_json(regions_path)['regions']
    if not regions:
        raise ResearchError('At least one named region is required.')
    for region in regions:
        bounds = region['box_xyxy']
        if (len(bounds) != 4 or any(type(v) is not int for v in bounds)
                or not 0 <= bounds[0] < bounds[2] <= target.shape[1]
                or not 0 <= bounds[1] < bounds[3] <= target.shape[0]
                or not region.get('name')):
            raise ResearchError('Region bounds must lie within the exact target.')
    cell_w, cell_h = 360, 230
    sizes = []
    for region in regions:
        x1, y1, x2, y2 = region['box_xyxy']
        scale = min((cell_w - 12) / (x2 - x1), cell_h / (y2 - y1))
        sizes.append((round((x2 - x1) * scale), round((y2 - y1) * scale)))
    canvas = Image.new('RGB', (cell_w * 3, 96 + sum(height + 34 for _, height in sizes)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), 'NEW PROPOSAL - exact target pixels enlarged; no new image detail or human approval', fill='black')
    draw.text((8, 23), 'Red: proposed building. Green: proposed scoring region. Yellow: newly added pixels.', fill='black')
    draw.text((8, 40), 'Magenta: removed pixels. Untinted areas are excluded from that mask; hidden walls are not drawn.', fill='black')
    for col, label in enumerate(('Original target', 'Building proposal + changes', 'Validity proposal + changes')):
        draw.text((col * cell_w + 8, 67), label, fill='black')
    y0 = 96
    for region, size in zip(regions, sizes):
        box = tuple(region['box_xyxy'])
        draw.text((8, y0), region['name'] + ' | target box ' + str(box), fill='black')
        for col, component in enumerate((None, 'building', 'valid')):
            pixels = target.copy()
            if component:
                tint = np.array([240, 45, 45] if component == 'building' else [20, 180, 80])
                mask, old = masks[component], old_masks[component]
                pixels[mask] = (.55 * pixels[mask] + .45 * tint).round().astype(np.uint8)
                pixels[mask & ~old] = (.35 * target[mask & ~old] + .65 * np.array([255, 220, 0])).round().astype(np.uint8)
                pixels[old & ~mask] = [240, 0, 220]
            tile = Image.fromarray(pixels).crop(box)
            tile = tile.resize(size, Image.Resampling.NEAREST)
            canvas.paste(tile, (col * cell_w + 6, y0 + 22))
        y0 += size[1] + 34
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    changes = {c: dict(added=int((masks[c] & ~old_masks[c]).sum()),
                       removed=int((old_masks[c] & ~masks[c]).sum())) for c in masks}
    save_json(out.with_suffix('.json'), dict(sample_id=sid, regions=regions,
        packet_sha256=sha256(root / 'packet.json'), previous_packet_sha256=sha256(parent / 'packet.json'),
        regions_sha256=sha256(regions_path), image_sha256=sha256(out), changes=changes,
        annotation_aid_only=True, masks_modified=False, approvals_recorded=False))
    print(changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'previous', 'sample-id', 'regions', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    preview(args.packet, args.previous, args.sample_id, args.regions, args.output)
