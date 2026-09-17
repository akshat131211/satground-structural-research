"""Open every image in one resumable review packet with Save & Next navigation."""
import argparse
from pathlib import Path
import urllib.request
import webbrowser

from satground.common import ResearchError, load_json, save_json
from satground.editor import create_server
from satground.editor_batch import BatchMaskEditorStore


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', type=int, help='Default: first unfinished image.')
    parser.add_argument('--packet', default='data/review/checkpoint-review18')
    parser.add_argument('--output', default='data/review/manual-edits')
    parser.add_argument('--approved-index')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    approved = args.approved_index
    if not approved and Path(args.packet).resolve() == Path('data/review/checkpoint-review18').resolve():
        public = Path('reports/ADDITIONAL_REVIEW_STATUS.json')
        if public.exists():
            count = load_json(public)['reviewed_count']
            approved = f'data/review/checkpoint-review18-approved/batch-index-v{count}.json' if count else None
    policy = Path('data/review/editor-policy.json')
    score = load_json(policy).get('score_building_additions', False) if policy.exists() else False
    if type(score) is not bool:
        raise ResearchError('The building-scoring preference must be true or false.')
    store = BatchMaskEditorStore(args.packet, args.output, approved, args.view,
                                 score_building_additions=score)
    info_path = store.output / 'server.json'
    try:
        store.acquire_writers()
    except ResearchError:
        if info_path.exists():
            url = load_json(info_path)['url']
            try:
                if not url.startswith('http://127.0.0.1:'):
                    raise ValueError('Not a loopback address.')
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status != 200:
                        raise ValueError('Editor unavailable.')
                if args.view:
                    url += '?view=' + str(args.view)
                print('Batch editor already running: ' + url, flush=True)
                if args.open_browser:
                    webbrowser.open(url)
                raise SystemExit(0)
            except (OSError, ValueError):
                pass
        raise
    try:
        server = create_server(store, args.port)
        url = f'http://127.0.0.1:{server.server_port}/'
        save_json(info_path, dict(url=url, packet=str(store.root), output=str(store.output),
                                  score_building_additions=score, batch=True))
        print(url, flush=True)
        print('All images available; existing drafts and accepted masks preserved.', flush=True)
        if args.open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        store.close()
