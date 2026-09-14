"""Fetch pinned public sky masks and retain only the training manifest's masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path, PurePosixPath
import zipfile

from huggingface_hub import hf_hub_download
from satground.common import (DATA_REPO, ResearchError, load_json, read_jsonl,
                              require_development, safe_data_path, save_json,
                              sha256, utc_now)


def import_city(city, rows, revision, data_root, download_root, report_dir):
    path = Path(hf_hub_download(repo_id=DATA_REPO, filename=city + '/pano_sky_mask.zip',
                               repo_type='dataset', revision=revision, local_dir=download_root))
    needed = {PurePosixPath(r['sky_mask']).name: r['sky_mask'] for r in rows if r['city'] == city}
    found, records = set(), []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if name.name not in needed:
                continue
            if ('..' in name.parts or name.is_absolute() or '\\' in info.filename
                    or ':' in info.filename or name.name in found):
                raise ResearchError('Unsafe or repeated sky-mask entry.')
            if not 0 < info.file_size < 16 * 1024 * 1024:
                raise ResearchError('Unexpected sky-mask size.')
            payload = archive.read(info)  # Also checks the selected member's ZIP CRC.
            target = safe_data_path(data_root, needed[name.name])
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if sha256(target) != digest:
                    raise ResearchError('Existing sky mask differs: ' + str(target))
            else:
                with target.open('xb') as output:
                    output.write(payload)
            found.add(name.name)
            records.append({'path': needed[name.name], 'sha256': digest})
    missing = sorted(set(needed) - found)
    record = {'city': city, 'revision': revision, 'archive_sha256': sha256(path),
              'selected': len(found), 'missing': missing, 'files': records,
              'created_utc': utc_now(), 'status': 'complete' if not missing else 'blocked'}
    save_json(Path(report_dir) / f'{city}-sky-masks.json', record)
    if missing:
        raise ResearchError(f'{city}: {len(missing)} missing training sky masks.')
    print(f'{city}: {len(found)} training masks verified.', flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifests/pilot/train.jsonl')
    p.add_argument('--data-root', default='data/vigor')
    p.add_argument('--download-root', default='data/supplements')
    p.add_argument('--report-dir', default='data/ingest')
    args = p.parse_args()
    rows = read_jsonl(args.manifest)
    require_development(rows, {'train'})
    revision = load_json('artifacts/asset-lock.json')['dataset_metadata_revision']
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(import_city, city, rows, revision, args.data_root,
                               args.download_root, args.report_dir)
                   for city in sorted({r['city'] for r in rows})]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise ResearchError('; '.join(errors))


if __name__ == '__main__':
    main()
