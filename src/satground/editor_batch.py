"""One review packet in one editor; completion markers never imply approval."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .annotations import verified_reviewed_masks
from .common import ResearchError, load_json, save_json, sha256, utc_now
from .editor import MaskEditorStore, unpack_mask


def writer_lock(directory):
    """Use the same per-draft OS lock as the single-image launcher."""
    handle = (Path(directory) / '.server.lock').open('a+b')
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise ResearchError('An editor is already writing this draft. Save and close its server first.')
    return handle


class BatchMaskEditorStore:
    def __init__(self, packet, output, approved_index=None, initial_view=None, *, score_building_additions=False):
        self.root = Path(packet).resolve()
        self.packet_hash = sha256(self.root / 'packet.json')
        samples = load_json(self.root / 'packet.json')['samples']
        numbers = [s['number'] for s in samples]
        ids = [s['sample_id'] for s in samples]
        if not samples or len(set(numbers)) != len(numbers) or len(set(ids)) != len(ids):
            raise ResearchError('Batch requires distinct numbered samples.')
        approved = load_json(approved_index) if approved_index else {}
        if not set(approved).issubset(ids):
            raise ResearchError('Approved index contains images outside this packet.')
        self.output = Path(output).resolve() / '_batches' / self.packet_hash[:12]
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stores, self.approved = {}, {}
        for sample in sorted(samples, key=lambda s: s['number']):
            sid, number = sample['sample_id'], sample['number']
            if sid in approved:
                entry = approved[sid]
                if entry['target_image_sha256'] != sample['target_image_sha256']:
                    raise ResearchError('Approved image differs from the batch target.')
                with Image.open(entry['target_image']) as image:
                    verified_reviewed_masks(entry, np.asarray(image.convert('RGB')))
                # Display the accepted packet, including assisted corrections, not an older draft.
                source = Path(entry['target_image']).parent.parent
                current = MaskEditorStore(source, output, sid, score_building_additions=score_building_additions)
                if any(current.sample[k + '_sha256'] != entry[k + '_sha256']
                       for k in ('target_image', 'building_mask', 'valid_mask')):
                    raise ResearchError('Approved source packet differs from the accepted masks.')
                self.approved[number] = entry
            else:
                current = MaskEditorStore(self.root, output, sid, score_building_additions=score_building_additions)
            self.stores[number] = current
        self.progress_path = self.output / 'progress.json'
        self.handles = []
        self.initial_view = initial_view
        if initial_view is not None:
            self.for_view(initial_view)
        else:
            self.initial_view = next((v['number'] for v in self.batch_state()['views']
                                      if v['status'] not in ('accepted', 'ready')), min(numbers))

    def acquire_writers(self):
        try:
            self.handles.append(writer_lock(self.output))
            for number, store in self.stores.items():
                if number not in self.approved:
                    self.handles.append(writer_lock(store.output))
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles = []

    def for_view(self, number=None):
        number = self.initial_view if number is None else number
        if number not in self.stores:
            raise ResearchError('Choose an image from this review batch.')
        return self.stores[number]

    def _progress(self):
        data = load_json(self.progress_path) if self.progress_path.exists() else dict(
            source_packet_sha256=self.packet_hash, completed={}, approval_recorded=False)
        if data['source_packet_sha256'] != self.packet_hash:
            raise ResearchError('Batch progress belongs to a different packet.')
        return data

    def batch_state(self):
        with self.lock:
            data, views = self._progress(), []
            for number, store in self.stores.items():
                store.verify_source()
                draft = store._read_draft()
                mark = data['completed'].get(str(number))
                ready = bool(mark and mark['draft_revision'] == draft['revision'] and
                             mark['masks'] == draft['masks'] and
                             sha256(Path(mark['output_directory']) / 'packet.json') == mark['packet_sha256'])
                status = 'accepted' if number in self.approved else 'ready' if ready else 'draft' if draft['saved_utc'] else 'pending'
                views.append(dict(number=number, status=status))
            accepted = sum(v['status'] == 'accepted' for v in views)
            ready_count = sum(v['status'] == 'ready' for v in views)
            return dict(views=views, total=len(views), accepted=accepted, ready=ready_count,
                        remaining=len(views)-accepted-ready_count, all_complete=accepted+ready_count == len(views))

    def state(self, number=None):
        store = self.for_view(number)
        state = store.state()
        number = store.sample['number']
        if number in self.approved:
            # Read-only accepted views always show the exact approved pixels.
            state['draft']['masks'] = state['original_masks']
            state['latest_version'] = None
        state.update(batch=self.batch_state(), read_only=number in self.approved)
        return state

    def save(self, number, payload, snapshot=False):
        with self.lock:
            store = self.for_view(number)
            number = store.sample['number']
            if number in self.approved:
                raise ResearchError('This accepted image is read-only; its approved masks are preserved.')
            if payload.get('sample_id') != store.sid:
                raise ResearchError('Save request belongs to a different image.')
            complete = payload.get('complete', False)
            if type(complete) is not bool or (complete and not snapshot):
                raise ResearchError('Completion requires Save & Next with both masks.')
            if complete and not unpack_mask(payload['masks']['valid'], store.target.shape[:2]).any():
                raise ResearchError('Mark some confidently assessable pixels in Validity before finishing this image.')
            result = store.save(payload, snapshot=snapshot)
            if complete:
                data = self._progress()
                data['completed'][str(number)] = dict(result, masks=store.draft['masks'],
                    sample_id=store.sid, completed_utc=utc_now(), interaction='save_and_next',
                    acceptance_pending=True)
                save_json(self.progress_path, data)
            result['batch'] = self.batch_state()
            return result
