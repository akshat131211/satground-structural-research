"""Verify completed edits exactly and measure the actual contour scoring support."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from verify_training_review import freeze_reviews, mask_measurements
from test_training_review import make_store, notes, masks
from satground.common import ResearchError, load_json, sha256
from satground.losses import soft_boundary


def completed_store(tmp_path):
    store = make_store(tmp_path)
    for number in (1, 2):
        review = notes(store, number)
        review.update(pairing='supported', support='mostly_inside', mask_review='checked')
        store.save_review(number, review)
        store.save(number, masks(store, number), snapshot=True)
    return store


def freeze(store, tmp_path):
    return freeze_reviews(store.root, tmp_path / 'edits', tmp_path / 'snapshot',
                          sha256(store.root / 'packet.json'), sha256(store.root / 'manifest.jsonl'), expected_count=2)


def test_exact_versions_freeze_without_promoting_to_training(tmp_path):
    store = completed_store(tmp_path)
    progress_before = sha256(store.progress_path)
    report = freeze(store, tmp_path)
    assert report['submitted_reviews'] == report['exact_mask_pairs_verified'] == 2
    assert report['human_supported_inside_checked'] == 2
    assert report['training_eligible_views'] == report['new_training_updates'] == 0
    assert sha256(store.progress_path) == progress_before
    record = load_json(tmp_path / 'snapshot/submissions.json')
    assert not record['human_submitted_masks_modified']
    assert all(not entry['training_eligible'] for entry in record['submissions'])
    with pytest.raises(ResearchError, match='fresh'):
        freeze(store, tmp_path)


def test_edits_after_completion_require_new_completion(tmp_path):
    store = completed_store(tmp_path)
    review = notes(store)
    review['evidence'] = 'Updated synthetic review remains uncertain.'
    store.save_review(1, review)
    with pytest.raises(ResearchError, match='Finish all current'):
        freeze(store, tmp_path)


def test_saved_png_tamper_is_rejected(tmp_path):
    store = completed_store(tmp_path)
    version = Path(store._progress()['completed']['1']['output_directory'])
    sample = load_json(version / 'packet.json')['samples'][0]
    path = version / sample['building_mask']
    path.write_bytes(path.read_bytes() + b'tamper')
    with pytest.raises(ResearchError, match='artifact changed'):
        freeze(store, tmp_path)


def test_context_history_tamper_is_rejected(tmp_path):
    store = completed_store(tmp_path)
    history = store.for_view(1).output / 'scene-review-history/revision-000001.json'
    history.write_text('{}', encoding='utf-8')
    with pytest.raises(ResearchError, match='history differs'):
        freeze(store, tmp_path)


def test_private_submission_cannot_be_exported_outside_review_directory(tmp_path):
    store = completed_store(tmp_path)
    with pytest.raises(ResearchError, match='private review drafts'):
        freeze_reviews(store.root, tmp_path / 'edits', tmp_path.parent / 'outside-review',
                       sha256(store.root / 'packet.json'), sha256(store.root / 'manifest.jsonl'), expected_count=2)


def test_contour_counts_match_training_operator_and_detect_ignored_edges():
    building = np.zeros((16, 16), dtype=bool)
    building[:8, :9] = True
    valid = np.ones_like(building)
    valid[7:9] = False
    valid[:, 8:10] = False
    result = mask_measurements(building, valid, building, valid)
    target = torch.from_numpy(building.astype(np.float32))[None, None]
    validity = torch.from_numpy(valid.astype(np.float32))[None, None]
    edge = soft_boundary(target) > 0
    interior = -torch.nn.functional.max_pool2d(-validity, 3, 1, 1) > 0
    assert result['raw_boundary_pixels'] == int(edge.sum()) > 0
    assert result['interior_valid_boundary_pixels'] == int((edge & interior).sum()) == 0
    assert result['changes']['building'] == {'added': 0, 'removed': 0}
