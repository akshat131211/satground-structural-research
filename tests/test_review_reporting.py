"""A reviewed-label report must not disguise prediction or cohort changes."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from satground.common import ResearchError, save_json, sha256, write_jsonl
from satground.review_reporting import report_reviewed_cohort, verify_label_only_change


def comparison_fixture():
    old = dict(count=3, generation_sha256='fixed-predictions', manifest_sha256='fixed-views',
        segmenter='independent', segmenter_revision='pinned', manually_reviewed_count=0)
    new = dict(old, manual_index_sha256='approved', manually_reviewed_count=2,
        mixed_manual_and_pseudo_labels=True)
    rows = [dict(sample_id=f'private-sample-{i}', geo_group='group', split='validation',
        label_provenance='independent_segmenter_pseudo_label', boundary_error=.2,
        building_iou=.5, lpips=.4, ssim=.6, psnr=10, valid_pixels=16) for i in range(3)]
    changed = deepcopy(rows)
    for r in changed[:2]:
        r.update(label_provenance='manually_reviewed', boundary_error=.1, building_iou=.7)
    ids = {r['sample_id'] for r in changed[:2]}
    return old, new, rows, changed, ids


def test_only_reviewed_structural_metrics_can_change():
    old, new, rows, changed, ids = comparison_fixture()
    assert len(verify_label_only_change(old, new, rows, changed, ids, 'approved')) == 3


@pytest.mark.parametrize('corruption', ['image_quality', 'unreviewed_structure', 'duplicate',
    'missing_view', 'predictions', 'index', 'label_provenance', 'group', 'segmenter', 'label_count'])
def test_rejects_changes_outside_review_scope(corruption):
    old, new, rows, changed, ids = comparison_fixture()
    if corruption == 'image_quality':
        changed[0]['lpips'] = .1
    elif corruption == 'unreviewed_structure':
        changed[2]['boundary_error'] = .1
    elif corruption == 'duplicate':
        changed.append(deepcopy(changed[0]))
    elif corruption == 'missing_view':
        changed.pop()
    elif corruption == 'predictions':
        new['generation_sha256'] = 'other-predictions'
    elif corruption == 'index':
        new['manual_index_sha256'] = 'other-index'
    elif corruption == 'label_provenance':
        changed[2]['label_provenance'] = 'manually_reviewed'
    elif corruption == 'group':
        changed[0]['geo_group'] = 'another-group'
    elif corruption == 'segmenter':
        new['segmenter_revision'] = 'other-revision'
    else:
        new['manually_reviewed_count'] = 1
    with pytest.raises(ResearchError):
        verify_label_only_change(old, new, rows, changed, ids, 'approved')


def test_report_keeps_per_view_coverage_without_private_identifiers(tmp_path):
    old, new, rows, changed, ids = comparison_fixture()
    index = {}
    for number, sid in enumerate(sorted(ids), 1):
        entry = dict(review_view_number=number, reviewed=True, reviewer='test-human',
            reviewed_utc='2026-09-16T10:00:00+00:00', annotation_method='human_reviewed_machine_proposal',
            proposal_provenance='test-fixture', review_blinded_to_model_predictions=False)
        for key in ('target_image', 'building_mask', 'valid_mask'):
            path = tmp_path / f'{sid}-{key}.png'
            pixels = np.zeros((4, 4, 3), np.uint8) if key == 'target_image' else np.full((4, 4), 255, np.uint8)
            Image.fromarray(pixels).save(path)
            entry[key], entry[key + '_sha256'] = str(path), sha256(path)
        index[sid] = entry
    index_path = tmp_path / 'index.json'
    save_json(index_path, index)
    new.update(manual_index_sha256=sha256(index_path), group_weighted={},
        evaluation_provenance=dict(pipeline_source_hash='same-source'))
    for name in ('GJ', 'GD'):
        for tag, summary, metrics in (('eval', old, rows), ('human2-eval', new, changed)):
            path = tmp_path / f'geometry100-{name}-validation-{tag}'
            save_json(path / 'summary.json', summary)
            write_jsonl(path / 'metrics.jsonl', metrics)
    out = tmp_path / 'report'
    result = report_reviewed_cohort(index_path, 'human2', out, tmp_path)
    assert result['reviewed_views'] == 2
    assert result['runs']['GD']['pseudo_label_views'] == 1
    assert all(v['validity_fraction'] == 1 for v in result['views'])
    assert not result['improvement_claim'] and not result['full_data_qa_complete']
    assert 'private-sample' not in json.dumps(result)
    with pytest.raises(ResearchError, match='fresh output'):
        report_reviewed_cohort(index_path, 'human2', out, tmp_path)
