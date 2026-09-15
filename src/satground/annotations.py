"""Exact-image human mask review; never infer approval from a pairing answer."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ResearchError, load_json, safe_data_path, save_json, sha256


def binary_mask(path, shape):
    with Image.open(path) as image:
        if image.mode != 'L':
            raise ResearchError('Reviewed masks must be single-channel L-mode PNGs.')
        array = np.asarray(image)
    if array.shape != tuple(shape) or not np.isin(array, (0, 255)).all():
        raise ResearchError('Reviewed masks must be exact-size binary 0/255 images.')
    return array == 255


def verified_reviewed_masks(entry, target):
    """Return masks only if their exact bytes and target pixels were reviewed."""
    if entry.get('reviewed') is not True or not entry.get('reviewer') or not entry.get('reviewed_utc'):
        raise ResearchError('Manual mask lacks explicit reviewer, date, and completed review.')
    paths = {k: Path(entry[k]) for k in ('building_mask', 'valid_mask', 'target_image')}
    for key, path in paths.items():
        digest = entry.get(key + '_sha256')
        if not digest or sha256(path) != digest:
            raise ResearchError('A reviewed target or mask changed after review: ' + key)
    with Image.open(paths['target_image']) as image:
        reviewed_target = np.asarray(image.convert('RGB'))
    if not np.array_equal(reviewed_target, target):
        raise ResearchError('Reviewed target pixels differ from the current evaluation crop.')
    masks = {k: binary_mask(path, target.shape[:2]) for k, path in paths.items() if k != 'target_image'}
    if not masks['valid_mask'].any():
        raise ResearchError('Reviewed validity mask contains no assessable pixels.')
    return masks['building_mask'], masks['valid_mask']


def finalize_review(packet, sample_id, decision_path, output):
    """Consume explicit human decisions for both displayed masks; no auto-approval.

    Decisions may also reject or request edits: those records remain local and
    cannot be exported as a reviewed evaluation index.
    """
    root = Path(packet).resolve()
    record = load_json(root / 'packet.json')
    decision = load_json(decision_path)
    if decision.get('sample_id') != sample_id or decision.get('packet_sha256') != sha256(root / 'packet.json'):
        raise ResearchError('Human decision belongs to a different sample or review packet.')
    sample = next((r for r in record['samples'] if r['sample_id'] == sample_id), None)
    if not sample:
        raise ResearchError('Sample is not in this review packet.')
    if not decision.get('reviewer') or not decision.get('reviewed_utc'):
        raise ResearchError('Record the actual human reviewer and review time.')
    try:
        timestamp = datetime.fromisoformat(decision['reviewed_utc'].replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('timezone absent')
    except (ValueError, TypeError) as error:
        raise ResearchError('Review time must be an explicit timezone-aware timestamp.') from error
    for component in ('building', 'valid'):
        answer = decision.get(component, {})
        if answer.get('decision') != 'accepted' or not answer.get('raw_human_answer'):
            raise ResearchError('Both exact masks need explicit human acceptance; pairing feedback is insufficient.')
        card = sample[component + '_card']
        if answer.get('card_sha256') != sample[component + '_card_sha256'] or sha256(safe_data_path(root, card)) != answer['card_sha256']:
            raise ResearchError('Human decision does not match the displayed mask card.')
    entry = dict(reviewed=True, reviewer=decision['reviewer'], reviewed_utc=decision['reviewed_utc'],
                 annotation_method='human_reviewed_machine_proposal',
                 proposal_model=record['proposal_model'], proposal_revision=record['proposal_revision'],
                 proposal_input_size=record.get('proposal_input_size'),
                 proposal_provenance=record.get('proposal_provenance', 'machine_pseudo_labels'),
                 correction_plan_sha256=record.get('correction_plan_sha256'),
                 review_blinded_to_model_predictions=record['review_blinded_to_model_predictions'],
                 decision_sha256=sha256(decision_path), instance_errors=None)
    for key in ('target_image', 'building_mask', 'valid_mask'):
        entry[key] = str(safe_data_path(root, sample[key]))
        entry[key + '_sha256'] = sample[key + '_sha256']
    with Image.open(entry['target_image']) as image:
        target = np.asarray(image.convert('RGB'))
    verified_reviewed_masks(entry, target)
    if Path(output).exists():
        raise ResearchError('Reviewed index already exists; export a new version.')
    save_json(output, {sample_id: entry})
    return dict(reviewed_masks_exported=1, output=str(output), full_dataset_qa_complete=False)
