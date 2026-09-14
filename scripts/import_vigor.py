"""Stream approved VIGOR city archives, extracting only a frozen manifest's RGB.

No archive is deleted. Seattle is rejected. gzip CRC and compressed SHA-256 are
checked through EOF, including bytes beyond the tar end marker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from satground.common import (ResearchError, load_json, read_jsonl,
                              require_development, safe_data_path, save_json,
                              sha256, utc_now, TRAIN_CITIES)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def import_archive(spec, rows, data_root, report_dir):
    city, kind = spec['city'], spec['kind']
    if city not in TRAIN_CITIES or kind not in ('satellite', 'panorama'):
        raise ResearchError('Only development-city RGB archives are allowed.')
    source = Path(spec['path']).resolve()
    before = source.stat()
    wanted = {str(PurePosixPath(r[kind]).relative_to(city)): r[kind]
              for r in rows if r['city'] == city}
    found, records = set(), []
    start, last_progress = time.monotonic(), 0
    count = 0
    with source.open('rb') as raw:
        hashed = HashingReader(raw)
        with gzip.GzipFile(fileobj=hashed) as uncompressed:
            with tarfile.open(fileobj=uncompressed, mode='r|') as archive:
                for member in archive:
                    name = member.name.removeprefix('./')
                    parts = PurePosixPath(name)
                    if (parts.is_absolute() or '..' in parts.parts or '\\' in name
                            or ':' in name or member.issym() or member.islnk()):
                        raise ResearchError('Unsafe archive member: ' + member.name)
                    if not member.isfile():
                        if member.isdir():
                            continue
                        raise ResearchError('Unsupported archive member: ' + member.name)
                    count += 1
                    if name in wanted:
                        if name in found or not 0 < member.size <= 32 * 1024 * 1024:
                            raise ResearchError('Repeated or oversized selected member: ' + name)
                        payload = archive.extractfile(member).read()
                        if len(payload) != member.size:
                            raise ResearchError('Incomplete selected member: ' + name)
                        digest = hashlib.sha256(payload).hexdigest()
                        target = safe_data_path(data_root, wanted[name])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            if sha256(target) != digest:
                                raise ResearchError('Existing data differs: ' + str(target))
                        else:
                            # Exclusive creation avoids overwriting previously imported data.
                            with target.open('xb') as output:
                                output.write(payload)
                        found.add(name)
                        records.append({'path': wanted[name], 'bytes': member.size, 'sha256': digest})
                    now = time.monotonic()
                    if now - last_progress > 15:
                        print(json.dumps({'archive': source.name, 'city': city, 'kind': kind,
                                          'read_percent': round(100 * raw.tell() / before.st_size, 1),
                                          'selected': len(found), 'required': len(wanted)}), flush=True)
                        last_progress = now
            while uncompressed.read(8 * 1024 * 1024):
                pass
        # GzipFile normally reads through compressed EOF; bind any remaining bytes too.
        while hashed.read(8 * 1024 * 1024):
            pass
        source_hash = hashed.digest.hexdigest()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ResearchError('Source archive changed during import: ' + source.name)
    missing = sorted(set(wanted) - found)
    report = {'archive': source.name, 'archive_bytes': before.st_size,
              'archive_sha256': source_hash, 'gzip_eof_verified': True,
              'city': city, 'kind': kind, 'members_scanned': count,
              'required': len(wanted), 'extracted': len(found), 'missing': missing,
              'files': records, 'elapsed_seconds': time.monotonic() - start,
              'created_utc': utc_now(), 'status': 'complete' if not missing else 'missing_members'}
    save_json(Path(report_dir) / f'{city}-{kind}.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'missing')}), flush=True)
    if missing:
        raise ResearchError(f'{source.name}: {len(missing)} required files absent.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping', required=True, help='Local JSON list of path/city/kind objects')
    p.add_argument('--manifest', default='data/manifests/pilot/all.jsonl')
    p.add_argument('--data-root', default='data/vigor')
    p.add_argument('--report-dir', default='data/ingest')
    p.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    args = p.parse_args()
    rows, specs = read_jsonl(args.manifest), load_json(args.mapping)
    require_development(rows)
    keys = [(s['city'], s['kind']) for s in specs]
    expected = {(r['city'], k) for r in rows for k in ('satellite', 'panorama')}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ResearchError('Supply exactly one archive for each required city/modality.')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(import_archive, s, rows, args.data_root, args.report_dir) for s in specs]
        reports, errors = [], []
        for future in futures:
            try:
                reports.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
    save_json(Path(args.report_dir) / 'summary.json', {
        'manifest_sha256': sha256(args.manifest), 'data_root': str(Path(args.data_root).resolve()),
        'created_utc': utc_now(), 'status': 'complete' if not errors else 'blocked',
        'errors': errors, 'selected_files': sum(r['extracted'] for r in reports),
        'archives_completed': len(reports), 'Seattle_extracted': False})
    if errors:
        raise ResearchError('; '.join(errors))


if __name__ == '__main__':
    main()
