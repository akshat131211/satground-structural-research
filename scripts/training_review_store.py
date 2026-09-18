"""Training context decisions alongside the existing mask editor; no auto-approval."""
import json
from pathlib import Path
from urllib.parse import urlsplit

from satground.common import ResearchError, load_json, read_jsonl, require_development, safe_data_path, save_json, sha256, utc_now
from satground.editor import DraftConflict, create_server
from satground.editor_batch import BatchMaskEditorStore

OPTIONS = dict(pairing={'pending','supported','uncertain','mismatch'},
    support={'pending','mostly_inside','partly_outside','outside','uncertain'},
    mask_review={'pending','checked','uncertain'})


class TrainingReviewStore(BatchMaskEditorStore):
    def __init__(self,packet,output,initial_view=None,*,score_building_additions=False):
        root=Path(packet)
        require_development(read_jsonl(root/'manifest.jsonl'),{'train'})
        if load_json(root/'packet.json').get('review_role')!='training_only':
            raise ResearchError('This interface requires a training-only packet.')
        super().__init__(packet,output,initial_view=initial_view,score_building_additions=score_building_additions)

    def review_state(self,number=None):
        store=self.for_view(number);path=store.output/'scene-review.json'
        state=load_json(path) if path.exists() else dict(revision=0,source_packet_sha256=self.packet_hash,
            sample_id=store.sid,pairing='pending',support='pending',mask_review='pending',reviewer='',evidence='',saved_utc=None)
        if state['sample_id']!=store.sid or state['source_packet_sha256']!=self.packet_hash:
            raise ResearchError('Context review belongs to another image.')
        return state

    def save_review(self,number,payload):
        with self.lock:
            store=self.for_view(number);store.verify_source();current=self.review_state(number)
            if payload.get('sample_id')!=store.sid or payload.get('source_packet_sha256')!=self.packet_hash:
                raise ResearchError('Context review belongs to another image.')
            if type(payload.get('expected_revision')) is not int or payload['expected_revision']!=current['revision']:
                raise DraftConflict('Review notes changed in another window. Reload before continuing.')
            for key,options in OPTIONS.items():
                if payload.get(key) not in options:raise ResearchError('Choose a valid '+key+' answer.')
            for key,limit in (('reviewer',120),('evidence',3000)):
                if not isinstance(payload.get(key),str) or len(payload[key])>limit:raise ResearchError('Invalid '+key+' text.')
            state=dict(revision=current['revision']+1,source_packet_sha256=self.packet_hash,sample_id=store.sid,
                **{k:payload[k] for k in (*OPTIONS,'reviewer','evidence')},saved_utc=utc_now(),
                interaction='local_training_context_form',approval_recorded=False,training_eligible=False)
            history=store.output/'scene-review-history'/f'revision-{state["revision"]:06d}.json'
            if history.exists():raise ResearchError('Preserve the existing context review revision.')
            save_json(history,state);save_json(store.output/'scene-review.json',state)
            return dict(review=state,batch=self.batch_state())

    @staticmethod
    def review_complete(review):
        return all(review[k]!='pending' for k in OPTIONS) and bool(review['reviewer'].strip()) and len(review['evidence'].strip())>=10

    def batch_state(self):
        with self.lock:
            result=super().batch_state();progress=self._progress()
            for view in result['views']:
                review=self.review_state(view['number']);mark=progress['completed'].get(str(view['number']),{})
                if view['status']=='ready' and (not self.review_complete(review) or mark.get('context_revision')!=review['revision']):
                    view['status']='draft'
            result['ready']=sum(v['status']=='ready' for v in result['views'])
            result['remaining']=result['total']-result['ready']-result['accepted']
            result['all_complete']=result['remaining']==0
            return result

    def state(self,number=None):
        state=super().state(number);state['training_review']=self.review_state(number)
        return state

    def save(self,number,payload,snapshot=False):
        with self.lock:
            review=self.review_state(number)
            if payload.get('complete') and not self.review_complete(review):
                raise ResearchError('Complete the context choices, reviewer name and a short evidence note before Save & Next. Uncertain answers are allowed.')
            result=super().save(number,payload,snapshot)
            if payload.get('complete'):
                store=self.for_view(number);progress=self._progress();mark=progress['completed'][str(store.sample['number'])]
                mark['context_revision']=review['revision'];mark['training_eligible']=False
                mark['context_sha256']=sha256(store.output/'scene-review.json')
                save_json(self.progress_path,progress);result['batch']=self.batch_state()
            return result


def create_training_server(store,port=0):
    server=create_server(store,port)
    base=server.RequestHandlerClass
    ui=Path(__file__).with_name('training_review_ui')

    class Handler(base):
        def reply(self,status,body,kind='application/json',filename=None):
            if kind.startswith('text/html') and urlsplit(self.path).path=='/':
                body=body.replace(b'</head>',b'<link rel="stylesheet" href="/training-review.css"><script src="/training-review.js" defer></script></head>')
                body=body.replace(b'<fieldset id="controls"', (ui/'panel.html').read_bytes()+b'<fieldset id="controls"')
            return super().reply(status,body,kind,filename)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path not in ('/training-review.js','/training-review.css','/training-context','/training-satellite'):
                return super().do_GET()
            if not self.authorized():return
            try:
                if path.startswith('/training-review.'):
                    file=ui/path[1:];kind='text/javascript' if path.endswith('.js') else 'text/css'
                else:
                    selected=store.for_view(self.selected_view());selected.verify_source()
                    key='context_image' if path=='/training-context' else 'satellite_image'
                    file=safe_data_path(store.root,selected.sample[key]);kind='image/png'
                    if sha256(file)!=selected.sample[key+'_sha256']:raise ResearchError('Context image changed.')
                self.reply(200,file.read_bytes(),kind)
            except ResearchError as error:self.reply(400,{'error':str(error)})

        def do_POST(self):
            if urlsplit(self.path).path!='/api/review':return super().do_POST()
            if not self.authorized(True):return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16000 or self.headers.get_content_type()!='application/json':
                    raise ResearchError('Expected bounded JSON review fields.')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ResearchError('Expected a review object.')
                self.reply(200,store.save_review(self.selected_view(),payload))
            except DraftConflict as error:self.reply(409,{'error':str(error)})
            except (ResearchError,ValueError,KeyError,TypeError) as error:self.reply(400,{'error':str(error)})

    server.RequestHandlerClass=Handler
    return server
