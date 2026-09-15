"""Human review is exact-image, component-specific, and invalidated by edits."""
import numpy as np
import pytest
from PIL import Image

from satground.annotations import binary_mask, finalize_review, verified_reviewed_masks
from satground.common import ResearchError, load_json, save_json, sha256


def packet(tmp_path):
    target = np.zeros((8, 8, 3), dtype=np.uint8)
    target[:, :, 0] = 120
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4, :4] = 255
    sample = dict(sample_id='one')
    for name, image in [('target_image', target), ('building_mask', mask), ('valid_mask', np.full((8, 8), 255, np.uint8)),
                        ('building_card', target), ('valid_card', target)]:
        path = tmp_path / f'{name}.png'
        Image.fromarray(image).save(path)
        sample[name], sample[name + '_sha256'] = path.name, sha256(path)
    record = dict(samples=[sample], proposal_model='machine', proposal_revision='pinned', review_blinded_to_model_predictions=False)
    save_json(tmp_path / 'packet.json', record)
    decision = dict(sample_id='one', packet_sha256=sha256(tmp_path / 'packet.json'), reviewer='test_fixture',
                    reviewed_utc='2026-09-15T10:00:00+00:00',
                    **{c: dict(decision='accepted', raw_human_answer='fixture explicit acceptance',
                               card_sha256=sample[c + '_card_sha256']) for c in ('building', 'valid')})
    save_json(tmp_path / 'decision.json', decision)
    return target, decision


def test_explicit_components_export_exact_reviewed_masks(tmp_path):
    target, decision = packet(tmp_path)
    result = finalize_review(tmp_path, 'one', tmp_path / 'decision.json', tmp_path / 'index.json')
    assert not result['full_dataset_qa_complete']
    entry = load_json(tmp_path / 'index.json')['one']
    building, valid = verified_reviewed_masks(entry, target)
    assert building.sum() == 16 and valid.all()
    assert entry['annotation_method'] == 'human_reviewed_machine_proposal'
    with pytest.raises(ResearchError, match='already exists'):
        finalize_review(tmp_path, 'one', tmp_path / 'decision.json', tmp_path / 'index.json')


@pytest.mark.parametrize('kind', ['pending_valid', 'pairing_answer', 'changed_card', 'wrong_packet'])
def test_rejects_incomplete_or_unrelated_decisions(tmp_path, kind):
    _, decision = packet(tmp_path)
    if kind == 'pending_valid':
        decision['valid']['decision'] = 'pending'
    elif kind == 'pairing_answer':
        decision['building']['decision'] = 'shared building clear'
    elif kind == 'changed_card':
        decision['building']['card_sha256'] = 'different'
    else:
        decision['packet_sha256'] = 'different'
    save_json(tmp_path / 'decision.json', decision)
    with pytest.raises(ResearchError):
        finalize_review(tmp_path, 'one', tmp_path / 'decision.json', tmp_path / 'index.json')
    assert not (tmp_path / 'index.json').exists()


def test_rejects_changed_mask_or_target_after_approval(tmp_path):
    target, _ = packet(tmp_path)
    finalize_review(tmp_path, 'one', tmp_path / 'decision.json', tmp_path / 'index.json')
    entry = load_json(tmp_path / 'index.json')['one']
    changed = target.copy()
    changed[0, 0, 1] = 1
    with pytest.raises(ResearchError, match='target pixels'):
        verified_reviewed_masks(entry, changed)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(entry['building_mask'])
    with pytest.raises(ResearchError, match='changed after review'):
        verified_reviewed_masks(entry, target)


def test_rejects_nonbinary_masks(tmp_path):
    image = np.full((8, 8), 127, np.uint8)
    path = tmp_path / 'not_binary.png'
    Image.fromarray(image).save(path)
    with pytest.raises(ResearchError, match='binary'):
        binary_mask(path, (8, 8))
