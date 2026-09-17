"""Audit whether the current training boundary mask retains positive targets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from satground.common import ResearchError, read_jsonl, require_development, sha256, utc_now
from satground.losses import soft_boundary
from satground.runs import read_config, validate_labels


def support_counts(building, valid):
    if building.shape != valid.shape or building.ndim != 2:
        raise ResearchError('Expected matching two-dimensional masks.')
    building = torch.as_tensor(np.asarray(building, dtype=np.float32))[None, None]
    valid = torch.as_tensor(np.asarray(valid, dtype=np.float32))[None, None]
    if not torch.all((building == 0) | (building == 1)) or not torch.all((valid == 0) | (valid == 1)):
        raise ResearchError('Expected binary masks.')
    target_boundary = soft_boundary(building) > 0
    interior = -F.max_pool2d(-valid, 3, stride=1, padding=1)
    scored_building = int((building * valid).sum())
    retained = int((target_boundary * interior).sum())
    return dict(raw_target_boundary_pixels=int(target_boundary.sum()),
                after_pixel_validity_pixels=int((target_boundary * valid).sum()),
                after_interior_validity_pixels=retained,
                scored_building_pixels=scored_building, valid_pixels=int(valid.sum()),
                interior_valid_pixels=int(interior.sum()),
                labelled_building_without_positive_boundary=scored_building > 0 and retained == 0)


def audit(output, local_output):
    output, local_output = Path(output), Path(local_output)
    if output.exists() or local_output.exists():
        raise ResearchError('Choose fresh audit output files.')
    cfg = read_config('configs/conservative500/A3.yaml')
    rows = read_jsonl(cfg['train_manifest']); require_development(rows, {'train'})
    validate_labels(cfg['labels'], cfg['train_manifest'], 'data/vigor')
    measured = {r['sample_id']: r for r in read_jsonl('runs/structural-gradients100/records.jsonl')
                if r['checkpoint_step'] == 0}
    records = []
    for row in rows:
        path = Path(cfg['labels']) / (row['sample_id'] + '.npz')
        with np.load(path) as labels:
            result = support_counts(labels['building'], labels['valid'])
        probe = measured[row['sample_id']]
        if result['after_interior_validity_pixels'] != probe['target_boundary_pixels']:
            raise ResearchError('CPU mask audit differs from gradient probe.')
        records.append(dict(sample_id=row['sample_id'], label_sha256=sha256(path), **result))
    local_output.parent.mkdir(parents=True, exist_ok=True)
    local_output.write_text(json.dumps(records, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    result = dict(created_utc=utc_now(), training_views=len(rows),
        labelled_nonempty_views=sum(r['scored_building_pixels'] > 0 for r in records),
        views_without_positive_boundary=sum(r['after_interior_validity_pixels'] == 0 for r in records),
        labelled_nonempty_views_without_positive_boundary=sum(r['labelled_building_without_positive_boundary'] for r in records),
        views_with_positive_boundary=sum(r['after_interior_validity_pixels'] > 0 for r in records),
        totals={k: sum(r[k] for r in records) for k in ('raw_target_boundary_pixels',
            'after_pixel_validity_pixels', 'after_interior_validity_pixels', 'valid_pixels', 'interior_valid_pixels')},
        exact_match_with_gradient_probe=True, labels_unchanged=True, loss_code_unchanged=True,
        manual_masks_used=False, local_records_sha256=sha256(local_output),
        training_manifest_sha256=sha256(cfg['train_manifest']),
        label_provenance_sha256=sha256(Path(cfg['labels']) / 'provenance.json'),
        interpretation='Most labelled nonempty training targets supply no positive boundary after current filtering; not a claim that pseudo boundaries are ground truth.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--local-output', required=True)
    a = p.parse_args(); print(json.dumps(audit(a.output, a.local_output), indent=2))
