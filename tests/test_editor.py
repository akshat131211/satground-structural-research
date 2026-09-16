"""The drawing tool preserves exact masks, resumability, and review boundaries."""
import io
import json
import re
import threading
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest
from PIL import Image

from satground.annotations import binary_mask
from satground.common import ResearchError, load_json, save_json, sha256, write_jsonl
from satground.editor import DraftConflict, MaskEditorStore, create_server, pack_mask, unpack_mask


def make_store(tmp_path):
    packet = tmp_path / 'source'
    packet.mkdir()
    sample = dict(sample_id='test-sample', number=1, reviewed=False)
    target = np.full((16, 16, 3), 90, np.uint8)
    for key, image in [('target_image', target), ('building_mask', np.zeros((16, 16), np.uint8)),
                       ('valid_mask', np.full((16, 16), 255, np.uint8))]:
        path = packet / (key + '.png')
        Image.fromarray(image).save(path)
        sample[key], sample[key + '_sha256'] = path.name, sha256(path)
    save_json(packet / 'packet.json', dict(samples=[sample], proposal_model='test-proposal',
        proposal_revision='fixture', human_review_complete=False, review_blinded_to_model_predictions=False))
    write_jsonl(packet / 'manifest.jsonl', [dict(sample_id='test-sample', split='validation')])
    return MaskEditorStore(packet, tmp_path / 'edits')


def payload(store):
    state = store.state()
    mask = unpack_mask(state['draft']['masks']['building'], (16, 16))
    mask[5:9, 7:12] = True
    return dict(source_packet_sha256=state['source_packet_sha256'], expected_revision=state['draft']['revision'],
                masks=dict(building=pack_mask(mask), valid=state['draft']['masks']['valid']))


def test_editor_saves_exact_resumable_unapproved_versions(tmp_path):
    store = make_store(tmp_path)
    original_hash = sha256(store.root / 'building_mask.png')
    first = store.save(payload(store), snapshot=True)
    path = store.output / 'versions' / first['version']
    packet = load_json(path / 'packet.json')
    sample = packet['samples'][0]
    assert binary_mask(path / sample['building_mask'], (16, 16)).sum() == 20
    assert binary_mask(path / sample['valid_mask'], (16, 16)).all()
    assert sha256(path / sample['target_image']) == store.sample['target_image_sha256']
    assert sha256(store.root / 'building_mask.png') == original_hash
    assert not first['approval_recorded'] and not packet['human_review_complete']
    decisions = load_json(path / 'decisions/test-sample.pending.json')
    assert decisions['building']['decision'] == decisions['valid']['decision'] == 'pending'
    resumed = MaskEditorStore(store.root, tmp_path / 'edits')
    assert resumed.state()['view_number'] == 1
    assert resumed.state()['draft']['revision'] == 1
    assert resumed.state()['latest_version']['draft_revision'] == 1
    assert unpack_mask(resumed.draft['masks']['building'], (16, 16)).sum() == 20
    second = resumed.save(payload(resumed), snapshot=True)
    assert first['version'] != second['version']
    assert sha256(path / 'packet.json') == first['packet_sha256']
    with zipfile.ZipFile(io.BytesIO(store.export_zip(first['version']))) as archive:
        with Image.open(io.BytesIO(archive.read(sample['building_mask']))) as image:
            assert image.mode == 'L' and image.size == (16, 16)
            assert set(np.asarray(image).ravel()) == {0, 255}


def test_editor_rejects_stale_wrong_source_and_malformed_saves(tmp_path):
    store = make_store(tmp_path)
    stale = payload(store)
    store.save(stale)
    with pytest.raises(DraftConflict):
        store.save(stale)
    wrong = payload(store)
    wrong['source_packet_sha256'] = 'different'
    with pytest.raises(ResearchError, match='different source'):
        store.save(wrong)
    bad = payload(store)
    bad['masks']['valid'] = 'bad!'
    with pytest.raises(ResearchError, match='encoding'):
        store.save(bad)
    assert store.state()['draft']['revision'] == 1
    with pytest.raises(ResearchError):
        store.export_zip('../../outside')
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(store.root / 'valid_mask.png')
    with pytest.raises(ResearchError, match='source target or mask changed'):
        store.save(payload(store))


def test_bit_packing_has_exact_size_and_padding():
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], bool)
    assert np.array_equal(mask, unpack_mask(pack_mask(mask), mask.shape))
    with pytest.raises(ResearchError, match='dimensions'):
        unpack_mask(pack_mask(mask), (32, 32))
    with pytest.raises(ResearchError, match='padding'):
        unpack_mask('//8=', (3, 3))


def test_loopback_api_rejects_cross_origin_and_unauthenticated_writes(tmp_path):
    store = make_store(tmp_path)
    server = create_server(store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base) as response:
            text = response.read().decode()
        token = re.search(r'name="editor-token" content="([^"]+)"', text)[1]
        headers = {'Content-Type': 'application/json', 'X-Mask-Editor-Token': token, 'Origin': base}
        body = json.dumps(payload(store)).encode()
        for overrides in ({'X-Mask-Editor-Token': 'wrong'}, {'Origin': 'https://unrelated.example'}, {'Host': 'unrelated.example'}):
            request = urllib.request.Request(base + '/api/draft', body, {**headers, **overrides}, method='POST')
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == 403
        assert store.state()['draft']['revision'] == 0
        request = urllib.request.Request(base + '/api/version', body, headers, method='POST')
        with urllib.request.urlopen(request) as response:
            saved = json.load(response)
        assert saved['revision'] == 1 and not saved['approval_recorded']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
