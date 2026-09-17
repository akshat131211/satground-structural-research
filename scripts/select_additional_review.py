"""Select additional validation review targets by geography, never by model scores."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

from satground.common import ResearchError, load_json, read_jsonl, require_development, sha256, utc_now, write_jsonl


def select(rows, accepted, count):
    require_development(rows, {'validation'})
    if count < 1 or not set(accepted).issubset({r['sample_id'] for r in rows}):
        raise ResearchError('Invalid selection size or accepted review IDs.')
    groups = defaultdict(deque)
    tiles = {r['satellite'] for r in rows if r['sample_id'] in accepted}
    panoramas = {r['panorama'] for r in rows if r['sample_id'] in accepted}
    for row in sorted(rows, key=lambda r: r['sample_id']):
        if row['sample_id'] not in accepted:
            groups[row['city'], row['geo_group']].append(row)
    cities = sorted({city for city, _ in groups})
    by_city = {c: sorted(k for k in groups if k[0] == c) for c in cities}
    order = [by_city[c][i] for i in range(max(map(len, by_city.values()), default=0))
             for c in cities if i < len(by_city[c])]
    chosen = []
    while len(chosen) < count:
        before = len(chosen)
        for key in order:
            queue = groups[key]
            while queue and (queue[0]['satellite'] in tiles or queue[0]['panorama'] in panoramas):
                queue.popleft()
            if queue:
                row = queue.popleft()
                chosen.append(row)
                tiles.add(row['satellite'])
                panoramas.add(row['panorama'])
                if len(chosen) == count:
                    break
        if len(chosen) == before:
            raise ResearchError('Insufficient independent candidate tiles/panoramas.')
    return chosen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/learning100/validation.jsonl')
    parser.add_argument('--accepted', default='data/review/manual-scored-first3/approved-index.json')
    parser.add_argument('--output', default='data/manifests/review/checkpoint-review18.jsonl')
    parser.add_argument('--count', type=int, default=18)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix('.selection.json').exists():
        raise ResearchError('Preserve previous selections; choose fresh output.')
    chosen = select(read_jsonl(args.manifest), load_json(args.accepted), args.count)
    write_jsonl(output, chosen)
    record = dict(created_utc=utc_now(), views=len(chosen), reviewed_count=0,
                  geographic_groups=len({r['geo_group'] for r in chosen}),
                  city_counts=dict(Counter(r['city'] for r in chosen)),
                  selection='City/geographic round-robin; sample-ID order within group; no predictions or scores consulted; distinct tiles and panoramas; accepted views excluded.',
                  source_manifest_sha256=sha256(args.manifest), accepted_index_sha256=sha256(args.accepted),
                  selected_manifest_sha256=sha256(output), audit_or_seattle_opened=False)
    output.with_suffix('.selection.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(record, indent=2))
