"""A proposed pixel edit cannot inherit human approval or alter its source."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from satground.common import ResearchError, load_json, save_json, sha256, write_jsonl

SCRIPTS = Path(__file__).parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('mask_patch', SCRIPTS / 'patch_mask_proposal.py')
patcher = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
spec.loader.exec_module(patcher)


def test_pixel_revision_is_immutable_unapproved_and_exact(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'revision'
    source.mkdir()
    sample = dict(sample_id='sample', number=1)
    for key, image in [('target_image', np.zeros((8, 8, 3), np.uint8)),
                       ('building_mask', np.zeros((8, 8), np.uint8)),
                       ('valid_mask', np.full((8, 8), 255, np.uint8))]:
        path = source / f'{key}.png'
        Image.fromarray(image).save(path)
        sample[key], sample[key + '_sha256'] = path.name, sha256(path)
    save_json(source / 'packet.json', dict(samples=[sample], reviewed_count=1, human_review_complete=True))
    write_jsonl(source / 'manifest.jsonl', [dict(sample_id='sample')])
    before = sha256(source / 'building_mask.png')
    plan = dict(sample_id='sample', source_packet_sha256=sha256(source / 'packet.json'),
                edits=[dict(component='building', operation='add', polygon_xy=[[1, 1], [3, 1], [3, 3], [1, 3]],
                            reason='Synthetic unit-test polygon')])
    save_json(tmp_path / 'plan.json', plan)
    patcher.patch(source, tmp_path / 'plan.json', out)
    assert sha256(source / 'building_mask.png') == before
    record = load_json(out / 'packet.json')
    assert not record['previous_approvals_inherited']
    assert not record['samples'][0]['reviewed']
    assert not record['human_review_complete']
    assert record['reviewed_count'] == 0
    mask = np.asarray(Image.open(out / record['samples'][0]['building_mask']))
    assert (mask == 255).sum() == 9
    decision = load_json(out / 'decisions/sample.pending.json')
    assert decision['building']['decision'] == decision['valid']['decision'] == 'pending'
    with pytest.raises(ResearchError, match='fresh revision'):
        patcher.patch(source, tmp_path / 'plan.json', out)
    plan['edits'][0]['polygon_xy'][0] = [-1, 0]
    save_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ResearchError, match='integer pixels'):
        patcher.patch(source, tmp_path / 'plan.json', tmp_path / 'invalid')
