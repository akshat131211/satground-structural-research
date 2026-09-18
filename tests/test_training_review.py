"""Keep training review separate and bind completion to the exact saved evidence."""
import json
import re
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from prepare_training_review import select_training
from training_review_store import TrainingReviewStore,create_training_server
from satground.common import ResearchError,load_json,save_json,sha256,write_jsonl
from satground.editor import DraftConflict


def test_selection_is_metadata_only_reproducible_and_geographically_distinct():
    rows=[dict(sample_id=f'{city}-{group}-{view}',geo_group=f'{city}-{group}',city=city,split='train',
               panorama=f'{city}/{group}/{view}',satellite=f'{city}/{group}.png')
          for city in ('Chicago','NewYork','SanFrancisco') for group in range(5) for view in range(2)]
    result=select_training(rows)
    assert result==select_training(list(reversed(rows)))
    assert len(result)==len({r['geo_group'] for r in result})==len({r['satellite'] for r in result})==12
    assert all(sum(r['city']==c for r in result)==4 for c in ('Chicago','NewYork','SanFrancisco'))
    rows[0]['split']='audit'
    with pytest.raises(ResearchError,match='not allowed'):select_training(rows)
    rows[0]['split']='train';rows[0]['city']='Seattle'
    with pytest.raises(ResearchError,match='sealed'):select_training(rows)
    with pytest.raises(ResearchError,match='four distinct'):select_training([r for r in result if r['city']!='Chicago'])


def make_store(tmp_path):
    root=tmp_path/'packet';root.mkdir();samples=[]
    for number in (1,2):
        s=dict(sample_id=f'test-{number}',number=number)
        for key,pixels in [('target_image',np.full((16,16,3),80,np.uint8)),
                           ('building_mask',np.zeros((16,16),np.uint8)),
                           ('valid_mask',np.full((16,16),255,np.uint8)),
                           ('context_image',np.full((32,32,3),120,np.uint8)),
                           ('satellite_image',np.full((32,32,3),60,np.uint8))]:
            path=root/f'{number}-{key}.png';Image.fromarray(pixels).save(path)
            s[key]=path.name;s[key+'_sha256']=sha256(path)
        samples.append(s)
    save_json(root/'packet.json',dict(samples=samples,review_role='training_only',proposal_model='synthetic-test',
        proposal_revision='fixture',human_review_complete=False,review_blinded_to_model_predictions=False))
    write_jsonl(root/'manifest.jsonl',[dict(sample_id=s['sample_id'],city='Chicago',split='train') for s in samples])
    return TrainingReviewStore(root,tmp_path/'edits')


def notes(store,number=1):
    s=store.state(number);r=s['training_review']
    return dict(sample_id=s['sample_id'],source_packet_sha256=s['source_packet_sha256'],expected_revision=r['revision'],
        pairing='uncertain',support='uncertain',mask_review='uncertain',reviewer='SYNTHETIC TEST',evidence='Synthetic unresolved correspondence.')


def masks(store,number=1,complete=True):
    s=store.state(number)
    return dict(sample_id=s['sample_id'],source_packet_sha256=s['source_packet_sha256'],
                expected_revision=s['draft']['revision'],masks=s['draft']['masks'],complete=complete)


def test_completion_requires_context_and_edits_invalidate_finished_revision(tmp_path):
    store=make_store(tmp_path)
    with pytest.raises(ResearchError,match='context choices'):store.save(1,masks(store),snapshot=True)
    first=notes(store);store.save_review(1,first)
    with pytest.raises(DraftConflict):store.save_review(1,first)
    with pytest.raises(ResearchError,match='another image'):store.save_review(2,notes(store,1))
    result=store.save(1,masks(store),snapshot=True)
    assert result['batch']['ready']==1 and result['approval_recorded'] is False
    progress=load_json(store.progress_path)['completed']['1']
    assert progress['training_eligible'] is False and progress['context_revision']==1
    saved_hash=sha256(Path(result['output_directory'])/'packet.json')
    changed=notes(store);changed['evidence']='Synthetic changed evidence remains uncertain.'
    store.save_review(1,changed)
    assert store.batch_state()['ready']==0
    resumed=TrainingReviewStore(store.root,tmp_path/'edits')
    assert resumed.review_state(1)['evidence']==changed['evidence']
    assert sha256(Path(result['output_directory'])/'packet.json')==saved_hash
    assert len(list((store.for_view(1).output/'scene-review-history').glob('*.json')))==2
    assert store.review_state(2)['revision']==0


def test_rejects_nontraining_packets_and_invalid_review_values(tmp_path):
    store=make_store(tmp_path)
    data=notes(store);data['pairing']='accepted_automatically'
    with pytest.raises(ResearchError,match='valid pairing'):store.save_review(1,data)
    assert store.review_state(1)['revision']==0
    write_jsonl(store.root/'manifest.jsonl',[dict(sample_id='test-1',city='Chicago',split='validation')])
    with pytest.raises(ResearchError,match='not allowed'):TrainingReviewStore(store.root,tmp_path/'edits')


def test_context_http_checks_origin_token_identity_and_asset_hash(tmp_path):
    store=make_store(tmp_path);server=create_training_server(store)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/?view=1') as response:html=response.read().decode()
        assert 'training-panel' in html and '/training-review.js' in html
        token=re.search(r'name="editor-token" content="([^"]+)"',html)[1]
        headers={'Content-Type':'application/json','X-Mask-Editor-Token':token}
        req=urllib.request.Request(base+'/api/review?view=1',json.dumps(notes(store)).encode(),headers)
        with urllib.request.urlopen(req) as response:assert json.load(response)['review']['revision']==1
        headers['Origin']='https://unrelated.invalid'
        with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(urllib.request.Request(base+'/api/review?view=1',json.dumps(notes(store)).encode(),headers))
        assert exc.value.code==403
        with urllib.request.urlopen(base+'/training-context?view=2') as response:assert response.headers['Content-Type']=='image/png'
        (store.root/store.for_view(2).sample['context_image']).write_bytes(b'changed')
        with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(base+'/training-context?view=2')
        assert exc.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)
