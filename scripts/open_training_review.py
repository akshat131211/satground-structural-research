"""Launch/reuse the local 12-view context and drawing interface."""
import argparse
from pathlib import Path
import urllib.request
import webbrowser

from satground.common import ResearchError,load_json,save_json
from training_review_store import TrainingReviewStore,create_training_server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',default='data/review/training12-v1')
    parser.add_argument('--output',default='data/review/training12-edits')
    parser.add_argument('--view',type=int);parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args();policy=load_json('data/review/editor-policy.json')
    score=policy.get('score_building_additions',False)
    if type(score) is not bool:raise ResearchError('Invalid building scoring preference.')
    store=TrainingReviewStore(args.packet,args.output,args.view,score_building_additions=score)
    info=store.output/'server.json'
    try:store.acquire_writers()
    except ResearchError:
        if info.exists():
            entry=load_json(info);url=entry['url']
            if entry.get('packet_sha256')==store.packet_hash and url.startswith('http://127.0.0.1:'):
                with urllib.request.urlopen(url,timeout=3) as response:
                    if response.status==200:
                        if args.view:url+='?view='+str(args.view)
                        print(url,flush=True)
                        if args.open_browser:webbrowser.open(url)
                        return
        raise
    try:
        server=create_training_server(store);url=f'http://127.0.0.1:{server.server_port}/'
        save_json(info,dict(url=url,packet_sha256=store.packet_hash,training_review=True,score_building_additions=score))
        print(url,flush=True)
        if args.open_browser:webbrowser.open(url)
        try:server.serve_forever()
        finally:server.server_close()
    finally:store.close()


if __name__=='__main__':main()
