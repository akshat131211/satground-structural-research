"""Verify pilot source records without certifying independent scene correspondence."""
import argparse
import json
from pathlib import Path

from satground.common import ResearchError, load_json, read_jsonl, require_development, sha256, utc_now
from satground.manifests import parse_pairs


def audit(output):
    output = Path(output)
    if output.exists():
        raise ResearchError('Preserve previous audits; use a fresh file.')
    metadata = Path('data/metadata/train__corrected_all_3city_remove_building.txt')
    if sha256(metadata) != load_json('data/manifests/pilot/split-report.json')['metadata_sha256']:
        raise ResearchError('Release pairing metadata changed.')
    pairs = {(r['satellite'], r['panorama']): r for r in parse_pairs(metadata)}
    counts = {}
    for split in ('train', 'validation'):
        rows = read_jsonl(f'data/manifests/learning100/{split}.jsonl')
        require_development(rows, {split})
        for row in rows:
            source = pairs.get((row['satellite'], row['panorama']))
            if source is None or any(row.get(k) != v for k, v in source.items()):
                raise ResearchError('Pilot view differs from its released source record.')
        counts[split] = len(rows)
    result = dict(created_utc=utc_now(), source_rows_checked=counts, all_match_released_metadata=True,
        metadata_sha256=sha256(metadata), train_manifest_sha256=sha256('data/manifests/learning100/train.jsonl'),
        validation_manifest_sha256=sha256('data/manifests/learning100/validation.jsonl'),
        independent_pairing_verified=False, metric_scale_verified=False, capture_dates_verified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args(); print(audit(a.output)['source_rows_checked'])
