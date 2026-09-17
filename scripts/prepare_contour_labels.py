"""Extend unchanged training pseudo-labels with separately supported contours."""
import argparse
import json
from pathlib import Path

import numpy as np

from satground.common import ResearchError, load_json, read_jsonl, require_development, sha256, utc_now
from satground.contours import DEFINITION, contour_support
from satground.runs import validate_labels


def prepare(output, report):
    output, report = Path(output), Path(report)
    if output.exists() or report.exists():
        raise ResearchError('Use fresh contour outputs; previous labels are immutable.')
    source = Path('data/labels/learning100')
    manifest = 'data/manifests/learning100/train.jsonl'
    rows = read_jsonl(manifest); require_development(rows, {'train'})
    provenance = validate_labels(source, manifest, 'data/vigor')
    if len(rows) != 100 or set(provenance['label_sha256']) != {r['sample_id'] for r in rows}:
        raise ResearchError('Expected unchanged complete 100-view training labels.')
    output.mkdir(parents=True)
    records, hashes = [], {}
    for row in rows:
        sid = row['sample_id']
        with np.load(source / (sid + '.npz')) as data:
            original = {k: data[k].copy() for k in data.files}
        support = contour_support(*(original[k] for k in ('building', 'valid', 'classes', 'confidence')))
        np.savez_compressed(output / (sid + '.npz'), **original, boundary_valid=support['boundary_valid'])
        with np.load(output / (sid + '.npz')) as written:
            if any(not np.array_equal(written[k], v) for k,v in original.items()):
                raise ResearchError('Original region labels changed during extension.')
        hashes[sid] = sha256(output / (sid + '.npz'))
        edge = support['target_boundary']
        records.append(dict(sample_id=sid, geo_group=row['geo_group'],
            nonempty=bool((original['building'] & original['valid']).any()),
            legacy_positive=int((edge & support['legacy_valid']).sum()),
            corrected_positive=int((edge & support['boundary_valid']).sum()),
            added_positive=int(support['added_positive'].sum()),
            original_valid_positions=int(support['legacy_valid'].sum()),
            corrected_valid_positions=int(support['boundary_valid'].sum())))
    local = output / 'support-records.json'
    local.write_text(json.dumps(records,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    provenance.update(boundary_definition=DEFINITION, label_sha256=hashes,
        source_label_provenance_sha256=sha256(source/'provenance.json'),
        region_arrays_unchanged=True, contour_human_verified=False)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    totals={k:sum(r[k] for r in records) for k in ('legacy_positive','corrected_positive','added_positive',
                                                'original_valid_positions','corrected_valid_positions')}
    nonempty=[r for r in records if r['nonempty']]
    covered=sum(r['corrected_positive'] > 0 for r in nonempty)
    # Feasibility only, fixed before measuring this rule; never a label-quality claim.
    ready=covered >= len(nonempty)/2 and totals['corrected_positive'] >= 1000
    result=dict(created_utc=utc_now(),definition=DEFINITION,training_views=len(rows),
        training_groups=len({r['geo_group'] for r in records}),nonempty_views=len(nonempty),
        nonempty_views_with_corrected_positive=covered,totals=totals,
        support_feasibility_passed=ready,minimum_positive_pixels=1000,
        minimum_nonempty_coverage_fraction=.5,human_verified=False,
        original_region_arrays_unchanged=True,reviewed_validation_masks_used=False,
        source_provenance_sha256=sha256(source/'provenance.json'),
        corrected_provenance_sha256=sha256(output/'provenance.json'),
        local_records_sha256=sha256(local),manifest_sha256=sha256(manifest),
        limitation='Anchored contours remain machine pseudo-labels, not certified building outlines.')
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    if not ready:
        raise ResearchError('Fixed contour rule failed target-support feasibility; do not train.')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/labels/contour100-v1')
    p.add_argument('--report',default='reports/contour500/target-support.json')
    a=p.parse_args();print(json.dumps(prepare(a.output,a.report),indent=2))
