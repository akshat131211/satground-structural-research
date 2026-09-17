import json
import re
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from satground.common import ResearchError, load_json, save_json, sha256, utc_now, write_jsonl
from satground.editor import DraftConflict, create_server, pack_mask, unpack_mask
from satground.editor_batch import BatchMaskEditorStore, writer_lock


def make_batch(tmp_path):
    root = tmp_path / 'packet'
    root.mkdir()
    samples = []
    for number in (1, 2):
        sample = dict(sample_id=f'test-{number}', number=number)
        for key, pixels in [('target_image', np.full((16, 16, 3), number * 60, np.uint8)),
                            ('building_mask', np.zeros((16, 16), np.uint8)),
                            ('valid_mask', np.full((16, 16), 255, np.uint8))]:
            path = root / f'{number}-{key}.png'
            Image.fromarray(pixels).save(path)
            sample[key], sample[key + '_sha256'] = path.name, sha256(path)
        samples.append(sample)
    save_json(root / 'packet.json', dict(samples=samples, proposal_model='synthetic-test',
        proposal_revision='fixture', human_review_complete=False, review_blinded_to_model_predictions=False))
    write_jsonl(root / 'manifest.jsonl', [dict(sample_id=s['sample_id'], split='validation') for s in samples])
    return BatchMaskEditorStore(root, tmp_path / 'edits')


def payload(batch, number, complete=False):
    state = batch.state(number)
    return dict(sample_id=state['sample_id'], source_packet_sha256=state['source_packet_sha256'],
                expected_revision=state['draft']['revision'], masks=state['draft']['masks'], complete=complete)


def test_finish_resume_edit_invalidation_and_no_cross_image_masks(tmp_path):
    batch = make_batch(tmp_path)
    first = payload(batch, 1, complete=True)
    mask = np.zeros((16, 16), bool)
    mask[2:4, 3:6] = True
    first['masks']['building'] = pack_mask(mask)
    # All images share a packet hash, so sample identity must be checked too.
    with pytest.raises(ResearchError, match='different image'):
        batch.save(2, first, snapshot=True)
    saved = batch.save(1, first, snapshot=True)
    assert saved['batch']['ready'] == 1 and not saved['approval_recorded']
    assert not unpack_mask(batch.state(2)['draft']['masks']['building'], (16, 16)).any()
    resumed = BatchMaskEditorStore(batch.root, tmp_path / 'edits')
    assert resumed.initial_view == 2 and resumed.batch_state()['ready'] == 1
    resumed.save(2, payload(resumed, 2), snapshot=True)
    assert resumed.batch_state()['ready'] == 1  # Save version alone is not Finished.
    resumed.save(2, payload(resumed, 2, complete=True), snapshot=True)
    assert resumed.batch_state()['all_complete']
    modified = payload(resumed, 1)
    modified['masks']['building'] = pack_mask(np.zeros((16, 16), bool))
    resumed.save(1, modified)
    assert resumed.batch_state()['remaining'] == 1
    assert sha256(saved['output_directory'] + '/packet.json') == saved['packet_sha256']
    with pytest.raises(DraftConflict):
        resumed.save(1, modified)


def test_completion_requires_nonempty_validity_and_snapshot(tmp_path):
    batch = make_batch(tmp_path)
    data = payload(batch, 1, complete=True)
    with pytest.raises(ResearchError, match='Completion requires'):
        batch.save(1, data)
    data['masks']['valid'] = pack_mask(np.zeros((16, 16), bool))
    with pytest.raises(ResearchError, match='assessable pixels'):
        batch.save(1, data, snapshot=True)
    assert batch.state(1)['draft']['revision'] == 0


def test_accepted_packet_overrides_old_draft_and_cannot_be_changed(tmp_path):
    batch = make_batch(tmp_path)
    data = payload(batch, 1)
    data['masks']['building'] = pack_mask(np.ones((16, 16), bool))
    result = batch.save(1, data, snapshot=True)
    root = batch.for_view(1).output / 'versions' / result['version']
    sample = load_json(root / 'packet.json')['samples'][0]
    entry = dict(reviewed=True, reviewer='synthetic_test_fixture', reviewed_utc=utc_now())
    for key in ('target_image', 'building_mask', 'valid_mask'):
        entry[key], entry[key + '_sha256'] = str(root / sample[key]), sample[key + '_sha256']
    approved = tmp_path / 'test-approved.json'
    save_json(approved, {'test-1': entry})
    data = payload(batch, 1)
    data['masks']['building'] = pack_mask(np.zeros((16, 16), bool))
    batch.save(1, data)
    reviewed = BatchMaskEditorStore(batch.root, tmp_path / 'edits', approved)
    state = reviewed.state(1)
    assert state['read_only'] and state['batch']['accepted'] == 1
    assert unpack_mask(state['draft']['masks']['building'], (16, 16)).all()
    with pytest.raises(ResearchError, match='read-only'):
        reviewed.save(1, payload(reviewed, 1), snapshot=True)


def test_batch_respects_existing_single_image_writer_lock(tmp_path):
    batch = make_batch(tmp_path)
    handle = writer_lock(batch.for_view(1).output)
    try:
        with pytest.raises(ResearchError, match='already writing'):
            batch.acquire_writers()
        assert batch.handles == []
    finally:
        handle.close()
    batch.acquire_writers()
    batch.close()


def test_http_routes_save_to_selected_image_only(tmp_path):
    batch = make_batch(tmp_path)
    server = create_server(batch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base + '/?view=2') as response:
            html = response.read().decode()
        token = re.search(r'name="editor-token" content="([^"]+)"', html)[1]
        headers = {'Content-Type': 'application/json', 'X-Mask-Editor-Token': token}
        req = urllib.request.Request(base + '/api/state?view=2', headers=headers)
        with urllib.request.urlopen(req) as response:
            state = json.load(response)
        assert state['view_number'] == 2 and state['batch']['total'] == 2
        request = urllib.request.Request(base + '/api/version?view=2',
                    json.dumps(payload(batch, 1, complete=True)).encode(), headers)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 400 and batch.state(2)['draft']['revision'] == 0
        request = urllib.request.Request(base + '/api/version?view=2',
                    json.dumps(payload(batch, 2, complete=True)).encode(), headers)
        with urllib.request.urlopen(request) as response:
            assert json.load(response)['batch']['ready'] == 1
        assert batch.state(1)['draft']['revision'] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
