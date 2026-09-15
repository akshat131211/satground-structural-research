"""Open the local, resumable binary-mask drawing editor."""
import argparse
import os
import webbrowser

from satground.common import load_json, save_json
from satground.editor import MaskEditorStore, create_server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', default='data/review/mask-first3-v7')
    parser.add_argument('--sample-id')
    parser.add_argument('--output', default='data/review/manual-edits')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    store = MaskEditorStore(args.packet, args.output, args.sample_id)
    # Keep one running writer per source draft, including double-click launches.
    lock = (store.output / '.server.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        info = store.output / 'server.json'
        if info.exists():
            url = load_json(info)['url']
            print('Editor already running: ' + url, flush=True)
            if args.open_browser and url.startswith('http://127.0.0.1:'):
                webbrowser.open(url)
        else:
            print('The editor is already starting. Use its existing window.', flush=True)
        raise SystemExit(0)
    server = create_server(store, args.port)
    url = f'http://127.0.0.1:{server.server_port}/'
    save_json(store.output / 'server.json', dict(url=url, packet=str(store.root), output=str(store.output)))
    print(url, flush=True)
    print('Drafts and saved versions: ' + str(store.output), flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        lock.close()
