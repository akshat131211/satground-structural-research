"""Prepare three target-only pseudo-mask proposals with separate human checks."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from satground.camera import crop_panorama
from satground.common import (ResearchError, read_jsonl, require_development, safe_data_path,
                              save_json, sha256, utc_now, write_jsonl)
from satground.semantics import EVAL_SEGMENTER, Segmenter, rgb_tensor


def review_card(target, mask, component, number):
    size = 384
    canvas = Image.new('RGB', (size * 3, size + 76), 'white')
    draw = ImageDraw.Draw(canvas)
    tint = np.array([240, 45, 45] if component == 'building' else [20, 180, 80])
    overlay = target.copy()
    overlay[mask] = (.55 * target[mask] + .45 * tint).round().astype(np.uint8)
    items = [Image.fromarray(target), Image.fromarray(overlay), Image.fromarray(mask.astype(np.uint8) * 255).convert('RGB')]
    labels = ['Real target image', 'Proposed building pixels: red' if component == 'building' else 'Proposed assessable pixels: green',
              'Exact mask: white=yes, black=no']
    for col, (image, label) in enumerate(zip(items, labels)):
        canvas.paste(image.resize((size, size), Image.Resampling.NEAREST), (col * size, 32))
        draw.text((col * size + 6, 9), label, fill='black')
    draw.text((8, size + 41), f'View {number}/3. MACHINE PROPOSAL: not reviewed. Inspect omissions and extra regions before accepting.', fill='black')
    draw.text((8, size + 58), 'Building: visible buildings only. Validity: assessable static pixels; exclude uncertain and dynamic regions.', fill='black')
    return canvas


def prepare(manifest, data_root, output):
    rows = read_jsonl(manifest)
    require_development(rows, {'validation'})
    rows = rows[:3]
    if len(rows) != 3:
        raise ResearchError('At least three preselected validation views are required.')
    out = Path(output)
    if out.exists():
        raise ResearchError('Mask packet exists; preserve it and use a fresh output directory.')
    for folder in ('targets', 'proposals', 'cards', 'decisions'):
        (out / folder).mkdir(parents=True)
    segmenter = Segmenter(EVAL_SEGMENTER)
    samples = []
    for number, row in enumerate(rows, 1):
        sid = row['sample_id']
        with Image.open(safe_data_path(data_root, row['panorama'])) as pano:
            target = crop_panorama(pano.convert('RGB'), row['yaw'], row['pitch'], row['fov'], row['size'])
        with torch.no_grad():
            certainty, labels = segmenter.probabilities(rgb_tensor(target, 'cuda')).max(1)
            labels, certainty = labels[0].cpu().numpy(), certainty[0].cpu().numpy()
        masks = dict(building=labels == 2, valid=(certainty >= .7) & (labels < 11))
        paths = dict(target_image=f'targets/{sid}.png')
        Image.fromarray(target).save(out / paths['target_image'])
        for component, mask in masks.items():
            paths[component + '_mask'] = f'proposals/{sid}-{component}.png'
            paths[component + '_card'] = f'cards/view-{number:02d}-{component}.jpg'
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / paths[component + '_mask'])
            review_card(target, mask, component, number).save(out / paths[component + '_card'], quality=96)
        sample = dict(sample_id=sid, number=number, reviewed=False, **paths,
            **{key + '_sha256': sha256(out / path) for key, path in paths.items()})
        samples.append(sample)
    write_jsonl(out / 'manifest.jsonl', rows)
    record = dict(created_utc=utc_now(), samples=samples,
        source_manifest_sha256=sha256(manifest), manifest_sha256=sha256(out / 'manifest.jsonl'),
        selection='First three views in existing learning100 validation order; no new content/metric selection.',
        purpose='Guided annotation feasibility only; not a replacement benchmark or main study.',
        proposal_model=segmenter.model_id, proposal_revision=segmenter.revision,
        proposal_input_size=segmenter.input_size,
        proposal_provenance='machine_pseudo_labels', human_review_complete=False, reviewed_count=0,
        review_blinded_to_model_predictions=False,
        blinding_note='Generated comparison galleries were previously shown for these development views. Cards here show only targets and proposals.',
        source_images_used_for_model_conditioning=False)
    save_json(out / 'packet.json', record)
    for sample in samples:
        decision = dict(sample_id=sample['sample_id'], packet_sha256=sha256(out / 'packet.json'),
            reviewer='', reviewed_utc='',
            **{component: dict(decision='pending', raw_human_answer='',
                card_sha256=sample[component + '_card_sha256']) for component in ('building', 'valid')})
        save_json(out / 'decisions' / f"{sample['sample_id']}.pending.json", decision)
    print(f'Prepared {len(samples)} unreviewed mask proposals and six separate cards at {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/learning100/validation.jsonl')
    parser.add_argument('--data-root', default='data/vigor')
    parser.add_argument('--output', default='data/review/mask-first3')
    args = parser.parse_args()
    prepare(args.manifest, args.data_root, args.output)
