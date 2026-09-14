"""Archive integrity matters: rejected inputs must never become ready data."""
import gzip
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

from satground.common import ResearchError, load_json, sha256

spec = importlib.util.spec_from_file_location('import_vigor', Path(__file__).parents[1] / 'scripts/import_vigor.py')
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


def archive(path, entries):
    with tarfile.open(path, 'w:gz') as stream:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))


def test_selective_import_preserves_bytes_and_verifies_compressed_hash(tmp_path):
    source = tmp_path / 'city.tar.gz'
    archive(source, [('satellite/keep.png', b'keep'), ('satellite/skip.png', b'skip')])
    row = {'city': 'Chicago', 'satellite': 'Chicago/satellite/keep.png'}
    result = ingest.import_archive({'city': 'Chicago', 'kind': 'satellite', 'path': source},
                                   [row], tmp_path / 'rgb', tmp_path / 'reports')
    assert result['archive_sha256'] == sha256(source)
    assert result['gzip_eof_verified'] and result['members_scanned'] == 2
    assert (tmp_path / 'rgb/Chicago/satellite/keep.png').read_bytes() == b'keep'
    assert not (tmp_path / 'rgb/Chicago/satellite/skip.png').exists()
    result2 = ingest.import_archive({'city': 'Chicago', 'kind': 'satellite', 'path': source},
                                    [row], tmp_path / 'rgb', tmp_path / 'reports')
    assert result2['status'] == 'complete'


def test_crc_failure_is_not_certified_as_complete(tmp_path):
    source = tmp_path / 'bad.tar.gz'
    archive(source, [('satellite/keep.png', b'data')])
    content = bytearray(source.read_bytes())
    content[-8] ^= 1  # Corrupt gzip CRC beyond the tar end marker.
    source.write_bytes(content)
    with pytest.raises((gzip.BadGzipFile, tarfile.ReadError)):
        ingest.import_archive({'city': 'Chicago', 'kind': 'satellite', 'path': source},
                              [{'city': 'Chicago', 'satellite': 'Chicago/satellite/keep.png'}],
                              tmp_path / 'rgb', tmp_path / 'reports')
    assert not (tmp_path / 'reports/Chicago-satellite.json').exists()


def test_seattle_and_archive_traversal_rejected(tmp_path):
    with pytest.raises(ResearchError, match='development'):
        ingest.import_archive({'city': 'Seattle', 'kind': 'satellite'}, [], tmp_path, tmp_path)
    source = tmp_path / 'escape.tar.gz'
    archive(source, [('../escape.png', b'bad')])
    with pytest.raises(ResearchError, match='Unsafe'):
        ingest.import_archive({'city': 'Chicago', 'kind': 'satellite', 'path': source},
                              [], tmp_path / 'rgb', tmp_path / 'reports')
    assert not (tmp_path / 'escape.png').exists()


def test_missing_manifest_members_are_reported(tmp_path):
    source = tmp_path / 'missing.tar.gz'
    archive(source, [('satellite/unrelated.png', b'data')])
    with pytest.raises(ResearchError, match='required files absent'):
        ingest.import_archive({'city': 'Chicago', 'kind': 'satellite', 'path': source},
                              [{'city': 'Chicago', 'satellite': 'Chicago/satellite/missing.png'}],
                              tmp_path / 'rgb', tmp_path / 'reports')
    assert load_json(tmp_path / 'reports/Chicago-satellite.json')['status'] == 'missing_members'
