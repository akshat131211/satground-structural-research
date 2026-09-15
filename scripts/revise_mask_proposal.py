"""Create a new version of a rejected proposal at a larger segmentation input.

This refines annotation candidates only. It does not replace past evaluation
labels, supply geometric truth, or inherit any prior human acceptance.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from satground.common import ResearchError, load_json, safe_data_path, save_json, sha256, utc_now, write_jsonl, read_jsonl
from satground.semantics import Segmenter, rgb_tensor
from prepare_mask_review import review_card


def revise(packet, sample_id, output, input_size=1024):
    root, out = Path(packet), Path(output)
    parent = load_json(root / 'packet.json')
    sample = next((s for s in parent['samples'] if s['sample_id'] == sample_id), None)
    if sample is None or input_size not in (512, 1024):
        raise ResearchError('Use a sample in the existing packet and a predefined review resolution.')
    if out.exists():
        raise ResearchError('Revision output already exists; preserve previous review records.')
    target_path = safe_data_path(root, sample['target_image'])
    if sha256(target_path) != sample['target_image_sha256']:
        raise ResearchError('Target changed after the original proposal.')
    with Image.open(target_path) as image:
        target = np.asarray(image.convert('RGB'))
    segmenter = Segmenter(parent['proposal_model'], parent['proposal_revision'], input_size=input_size)
    with torch.no_grad():
        certainty, labels = segmenter.probabilities(rgb_tensor(target, 'cuda')).max(1)
        labels, certainty = labels[0].cpu().numpy(), certainty[0].cpu().numpy()
    for name in ('targets', 'proposals', 'cards', 'decisions'):
        (out / name).mkdir(parents=True)
    paths = dict(target_image=f'targets/{sample_id}.png')
    Image.fromarray(target).save(out / paths['target_image'])
    masks = dict(building=labels == 2, valid=(certainty >= .7) & (labels < 11))
    for component, mask in masks.items():
        paths[component + '_mask'] = f'proposals/{sample_id}-{component}.png'
        paths[component + '_card'] = f'cards/view-{sample["number"]:02d}-{component}.jpg'
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / paths[component + '_mask'])
        review_card(target, mask, component, sample['number']).save(out / paths[component + '_card'], quality=96)
    revised = dict(sample_id=sample_id, number=sample['number'], reviewed=False, **paths,
        **{key + '_sha256': sha256(out / path) for key, path in paths.items()})
    row = next(r for r in read_jsonl(root / 'manifest.jsonl') if r['sample_id'] == sample_id)
    write_jsonl(out / 'manifest.jsonl', [row])
    record = dict(parent, created_utc=utc_now(), samples=[revised],
        human_review_complete=False, reviewed_count=0,
        source_packet_sha256=sha256(root / 'packet.json'), manifest_sha256=sha256(out / 'manifest.jsonl'),
        proposal_input_size=input_size, purpose='Revision of rejected annotation proposal; no model-quality comparison.',
        selection='Previously selected view 1; revised after user rejected the displayed building mask.',
        previous_approvals_inherited=False)
    save_json(out / 'packet.json', record)
    save_json(out / 'decisions' / f'{sample_id}.pending.json', dict(sample_id=sample_id,
        packet_sha256=sha256(out / 'packet.json'), reviewer='', reviewed_utc='',
        **{c: dict(decision='pending', raw_human_answer='', card_sha256=revised[c + '_card_sha256']) for c in masks}))
    print('New proposal requires new human review:', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'sample-id', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--input-size', type=int, choices=[512, 1024], default=1024)
    args = parser.parse_args()
    revise(args.packet, args.sample_id, args.output, args.input_size)
