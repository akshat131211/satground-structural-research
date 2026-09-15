"""Apply explicit pixel-coordinate annotation proposals, preserving versions.

An edit plan names the inspected target/mask hashes and additive/subtractive
polygons. These are assistant proposals requiring human review, never automatic
ground-truth annotations.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from satground.annotations import binary_mask
from satground.common import ResearchError, load_json, read_jsonl, safe_data_path, save_json, sha256, utc_now, write_jsonl
from prepare_mask_review import review_card


def patch(packet, plan_path, output):
    root, out = Path(packet), Path(output)
    parent, plan = load_json(root / 'packet.json'), load_json(plan_path)
    if out.exists():
        raise ResearchError('Choose a fresh revision output.')
    sample = next((s for s in parent['samples'] if s['sample_id'] == plan['sample_id']), None)
    if sample is None or plan.get('source_packet_sha256') != sha256(root / 'packet.json'):
        raise ResearchError('Correction plan belongs to another review version.')
    target_path = safe_data_path(root, sample['target_image'])
    if sha256(target_path) != sample['target_image_sha256']:
        raise ResearchError('Annotation target changed.')
    with Image.open(target_path) as image:
        target = np.asarray(image.convert('RGB'))
    masks = {}
    for component in ('building', 'valid'):
        path = safe_data_path(root, sample[component + '_mask'])
        if sha256(path) != sample[component + '_mask_sha256']:
            raise ResearchError('Source mask changed.')
        masks[component] = binary_mask(path, target.shape[:2])
    for edit in plan['edits']:
        component, points = edit['component'], edit['polygon_xy']
        if component not in masks or edit['operation'] not in ('add', 'remove') or not edit.get('reason'):
            raise ResearchError('Every explicit correction requires a mask, operation, and reason.')
        if len(points) < 3 or any(len(p) != 2 or any(type(v) is not int for v in p)
                or not (0 <= p[0] < target.shape[1] and 0 <= p[1] < target.shape[0]) for p in points):
            raise ResearchError('Correction polygons must use integer pixels within the exact target.')
        image = Image.fromarray(masks[component].astype(np.uint8) * 255)
        ImageDraw.Draw(image).polygon([tuple(p) for p in points], fill=255 if edit['operation'] == 'add' else 0)
        masks[component] = np.asarray(image) == 255
    for name in ('targets', 'proposals', 'cards', 'decisions'):
        (out / name).mkdir(parents=True)
    sid = sample['sample_id']
    paths = dict(target_image=f'targets/{sid}.png')
    Image.fromarray(target).save(out / paths['target_image'])
    for component, mask in masks.items():
        paths[component + '_mask'] = f'proposals/{sid}-{component}.png'
        paths[component + '_card'] = f'cards/view-{sample["number"]:02d}-{component}.jpg'
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / paths[component + '_mask'])
        review_card(target, mask, component, sample['number']).save(out / paths[component + '_card'], quality=96)
    revised = dict(sample_id=sid, number=sample['number'], reviewed=False, **paths,
        **{key + '_sha256': sha256(out / path) for key, path in paths.items()})
    rows = [r for r in read_jsonl(root / 'manifest.jsonl') if r['sample_id'] == sid]
    write_jsonl(out / 'manifest.jsonl', rows)
    save_json(out / 'correction-plan.json', plan)
    record = dict(parent, created_utc=utc_now(), samples=[revised],
        human_review_complete=False, reviewed_count=0,
        source_packet_sha256=sha256(root / 'packet.json'), manifest_sha256=sha256(out / 'manifest.jsonl'),
        proposal_provenance='machine_labels_with_explicit_assistant_pixel_corrections_pending_human_review',
        correction_plan_sha256=sha256(out / 'correction-plan.json'), previous_approvals_inherited=False)
    save_json(out / 'packet.json', record)
    save_json(out / 'decisions' / f'{sid}.pending.json', dict(sample_id=sid,
        packet_sha256=sha256(out / 'packet.json'), reviewer='', reviewed_utc='',
        **{c: dict(decision='pending', raw_human_answer='', card_sha256=revised[c + '_card_sha256']) for c in masks}))
    print('Saved an unapproved pixel-correction proposal:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'plan', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    patch(args.packet, args.plan, args.output)
