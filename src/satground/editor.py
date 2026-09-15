"""Local binary-mask editor storage and loopback HTTP interface.

Drafts can be resumed. Saved versions are immutable proposals, never implicit
human approval or training eligibility. The service exposes no dataset browser.
"""
from __future__ import annotations

import base64
import binascii
import copy
import io
import json
import secrets
import threading
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw

from .annotations import binary_mask
from .common import ResearchError, load_json, read_jsonl, safe_data_path, save_json, sha256, utc_now, write_jsonl


class DraftConflict(ResearchError):
    pass


def pack_mask(mask):
    return base64.b64encode(np.packbits(np.asarray(mask, bool).ravel(), bitorder='little')).decode('ascii')


def unpack_mask(encoded, shape):
    count = int(np.prod(shape))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError, binascii.Error) as error:
        raise ResearchError('Invalid mask encoding.') from error
    if len(raw) != (count + 7) // 8:
        raise ResearchError('Mask data has the wrong dimensions.')
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder='little')
    if bits[count:].any():
        raise ResearchError('Nonzero mask padding is invalid.')
    return bits[:count].reshape(shape).astype(bool)


def mask_card(target, mask, component):
    size = 384
    canvas = Image.new('RGB', (size * 3, size + 78), 'white')
    draw = ImageDraw.Draw(canvas)
    tint = np.array([240, 45, 45] if component == 'building' else [20, 180, 80])
    overlay = target.copy()
    overlay[mask] = (.55 * target[mask] + .45 * tint).round().astype(np.uint8)
    labels = ('Exact original target', 'Building: red' if component == 'building' else 'Validity: green',
              'Binary mask: white=included')
    for i, (pixels, label) in enumerate(zip((target, overlay, mask.astype(np.uint8) * 255), labels)):
        image = Image.fromarray(pixels).convert('RGB').resize((size, size), Image.Resampling.NEAREST)
        canvas.paste(image, (i * size, 32))
        draw.text((i * size + 6, 9), label, fill='black')
    draw.text((8, size + 42), 'Editor-saved proposal. Saving is not final human acceptance.', fill='black')
    draw.text((8, size + 59), 'Building and validity are separate masks; uncertain pixels can remain unscored.', fill='black')
    return canvas


class MaskEditorStore:
    def __init__(self, packet, output, sample_id=None):
        self.root, self.output = Path(packet).resolve(), Path(output).resolve()
        self.record = load_json(self.root / 'packet.json')
        samples = self.record['samples']
        matches = [s for s in samples if s['sample_id'] == sample_id] if sample_id else samples
        if len(matches) != 1:
            raise ResearchError('Specify exactly one sample to edit.')
        self.sample = matches[0]
        self.sid = self.sample['sample_id']
        self.packet_hash = sha256(self.root / 'packet.json')
        self.output = self.output / self.sid / self.packet_hash[:12]
        self.lock = threading.RLock()
        self.verify_source()
        with Image.open(safe_data_path(self.root, self.sample['target_image'])) as image:
            self.target = np.asarray(image.convert('RGB'))
        self.original = {c: binary_mask(safe_data_path(self.root, self.sample[c + '_mask']), self.target.shape[:2])
                         for c in ('building', 'valid')}
        self.output.mkdir(parents=True, exist_ok=True)
        self.draft_path = self.output / 'draft.json'
        self.draft = self._read_draft()

    def verify_source(self):
        if sha256(self.root / 'packet.json') != self.packet_hash:
            raise ResearchError('The source proposal changed. Restart with its new version.')
        for key in ('target_image', 'building_mask', 'valid_mask'):
            if sha256(safe_data_path(self.root, self.sample[key])) != self.sample[key + '_sha256']:
                raise ResearchError('The source target or mask changed.')

    def _read_draft(self):
        if self.draft_path.exists():
            draft = load_json(self.draft_path)
            if draft.get('source_packet_sha256') != self.packet_hash:
                raise ResearchError('Draft belongs to another source packet.')
            for c in ('building', 'valid'):
                unpack_mask(draft['masks'][c], self.target.shape[:2])
            return draft
        return dict(revision=0, saved_utc=None, source_packet_sha256=self.packet_hash,
                    masks={c: pack_mask(m) for c, m in self.original.items()}, approval_recorded=False)

    def state(self):
        with self.lock:
            self.verify_source()
            self.draft = self._read_draft()
            source = safe_data_path(self.root, self.sample['target_image'])
            return dict(sample_id=self.sid, width=self.target.shape[1], height=self.target.shape[0],
                        target='data:image/png;base64,' + base64.b64encode(source.read_bytes()).decode('ascii'),
                        source_packet_sha256=self.packet_hash, target_sha256=self.sample['target_image_sha256'],
                        original_masks={c: pack_mask(m) for c, m in self.original.items()},
                        draft=copy.deepcopy(self.draft), output_directory=str(self.output),
                        latest_version=load_json(self.output / 'latest-version.json') if (self.output / 'latest-version.json').exists() else None,
                        approval_recorded=False)

    def save(self, payload, snapshot=False):
        with self.lock:
            self.verify_source()
            self.draft = self._read_draft()
            if payload.get('source_packet_sha256') != self.packet_hash:
                raise ResearchError('Save request belongs to a different source image.')
            if type(payload.get('expected_revision')) is not int or payload['expected_revision'] != self.draft['revision']:
                raise DraftConflict('Another window saved a newer draft. Your local edits are still on screen; export the backup before reloading.')
            if set(payload.get('masks', {})) != {'building', 'valid'}:
                raise ResearchError('Both building and validity masks are required.')
            masks = {c: unpack_mask(payload['masks'][c], self.target.shape[:2]) for c in ('building', 'valid')}
            draft = dict(revision=self.draft['revision'] + 1, saved_utc=utc_now(), source_packet_sha256=self.packet_hash,
                         masks={c: pack_mask(m) for c, m in masks.items()}, approval_recorded=False)
            save_json(self.draft_path, draft)
            self.draft = draft
            result = dict(revision=draft['revision'], saved_utc=draft['saved_utc'], approval_recorded=False)
            if snapshot:
                result.update(self._snapshot(masks, draft))
            return result

    def _snapshot(self, masks, draft):
        name = 'edit-' + uuid.uuid4().hex
        out = self.output / 'versions' / name
        out.mkdir(parents=True, exist_ok=False)
        for directory in ('targets', 'proposals', 'cards', 'decisions'):
            (out / directory).mkdir()
        paths = dict(target_image=f'targets/{self.sid}.png')
        (out / paths['target_image']).write_bytes(safe_data_path(self.root, self.sample['target_image']).read_bytes())
        for c, mask in masks.items():
            paths[c + '_mask'] = f'proposals/{self.sid}-{c}.png'
            paths[c + '_card'] = f'cards/view-{self.sample["number"]:02d}-{c}.jpg'
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / paths[c + '_mask'])
            mask_card(self.target, mask, c).save(out / paths[c + '_card'], quality=96)
        sample = dict(sample_id=self.sid, number=self.sample['number'], reviewed=False, **paths,
                      **{key + '_sha256': sha256(out / path) for key, path in paths.items()})
        rows = [r for r in read_jsonl(self.root / 'manifest.jsonl') if r['sample_id'] == self.sid]
        write_jsonl(out / 'manifest.jsonl', rows)
        changes = {c: dict(added=int((m & ~self.original[c]).sum()), removed=int((~m & self.original[c]).sum()))
                   for c, m in masks.items()}
        edit = dict(saved_utc=draft['saved_utc'], source_packet_sha256=self.packet_hash, draft_revision=draft['revision'],
                    changes=changes, interaction='local_binary_mask_editor', operator_identity_not_inferred=True,
                    human_approved=False, target_changed=False)
        save_json(out / 'editor-changes.json', edit)
        record = dict(self.record, created_utc=utc_now(), samples=[sample], manifest_sha256=sha256(out / 'manifest.jsonl'),
                      human_review_complete=False, reviewed_count=0, previous_approvals_inherited=False,
                      source_packet_sha256=self.packet_hash,
                      proposal_provenance='machine_proposal_with_local_editor_changes_pending_human_review',
                      correction_plan_sha256=sha256(out / 'editor-changes.json'))
        save_json(out / 'packet.json', record)
        save_json(out / 'decisions' / f'{self.sid}.pending.json', dict(sample_id=self.sid,
            packet_sha256=sha256(out / 'packet.json'), reviewer='', reviewed_utc='',
            **{c: dict(decision='pending', raw_human_answer='', card_sha256=sample[c + '_card_sha256']) for c in masks}))
        result = dict(version=name, output_directory=str(out), changes=changes, draft_revision=draft['revision'],
                      packet_sha256=sha256(out / 'packet.json'), approval_recorded=False)
        save_json(self.output / 'latest-version.json', result)
        return result

    def export_zip(self, version):
        if not version.startswith('edit-') or len(version) != 37 or any(c not in '0123456789abcdef' for c in version[5:]):
            raise ResearchError('Unknown saved version.')
        path = self.output / 'versions' / version
        if not (path / 'packet.json').is_file():
            raise ResearchError('Saved version not found.')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(path.rglob('*')):
                if file.is_file() and not file.is_symlink():
                    archive.write(file, str(file.relative_to(path)))
        return stream.getvalue()


def create_server(store, port=0):
    token = secrets.token_urlsafe(32)
    ui = Path(__file__).with_name('editor_ui')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, body, kind='application/json', filename=None):
            if kind == 'application/json':
                body = json.dumps(body, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self, api=False):
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != host:
                self.reply(403, {'error': 'Use the printed loopback URL.'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + host:
                self.reply(403, {'error': 'Cross-origin access is not allowed.'})
                return False
            if api and not secrets.compare_digest(self.headers.get('X-Mask-Editor-Token', ''), token):
                self.reply(403, {'error': 'Editor session token missing or stale.'})
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path).path
            if not self.authorized(path.startswith('/api/')):
                return
            try:
                if path == '/api/state':
                    self.reply(200, store.state())
                elif path.startswith('/api/export/'):
                    version = path.removeprefix('/api/export/')
                    self.reply(200, store.export_zip(version), 'application/zip', 'satground-' + version + '.zip')
                elif path in ('/', '/editor.js', '/editor-core.js', '/editor.css'):
                    file = ui / ('index.html' if path == '/' else path[1:])
                    content = file.read_bytes()
                    if path == '/':
                        content = content.replace(b'__EDITOR_TOKEN__', token.encode('ascii'))
                    kind = 'text/html; charset=utf-8' if path == '/' else 'text/css; charset=utf-8' if path.endswith('.css') else 'text/javascript; charset=utf-8'
                    self.reply(200, content, kind)
                else:
                    self.reply(404, {'error': 'Not found.'})
            except ResearchError as error:
                self.reply(400, {'error': str(error)})

        def do_POST(self):
            if not self.authorized(True):
                return
            path = urlsplit(self.path).path
            if path not in ('/api/draft', '/api/version'):
                self.reply(404, {'error': 'Not found.'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 512000 or self.headers.get_content_type() != 'application/json':
                    raise ResearchError('Expected a bounded JSON mask request.')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ResearchError('Expected a JSON object.')
                self.reply(200, store.save(payload, snapshot=path == '/api/version'))
            except DraftConflict as error:
                self.reply(409, {'error': str(error)})
            except (ResearchError, ValueError, KeyError, TypeError) as error:
                self.reply(400, {'error': str(error)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)
