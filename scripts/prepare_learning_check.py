"""Select a fixed preliminary subset before looking at predictions or metrics."""
from pathlib import Path

import yaml
from satground.common import (ResearchError, read_jsonl, require_development,
                              save_json, sha256, utc_now, write_jsonl)
from satground.manifests import assert_disjoint


def main():
    selected = []
    for split, count in [('train', 100), ('validation', 24)]:
        rows = read_jsonl(f'data/manifests/pilot/{split}.jsonl')
        require_development(rows, {split})
        if len(rows) < count:
            raise ResearchError('Insufficient source rows for the learning check.')
        rows = sorted(rows, key=lambda r: r['sample_id'])[:count]
        selected.extend(rows)
        destination = Path(f'data/manifests/learning100/{split}.jsonl')
        if destination.exists() and read_jsonl(destination) != rows:
            raise ResearchError('Existing learning subset differs; do not overwrite it.')
        write_jsonl(destination, rows)
    assert_disjoint(selected)
    save_json('data/manifests/learning100/selection.json', {
        'selection': 'first sample-ID-sorted rows of pre-existing manifests; no content/metric selection',
        'train_rows': 100, 'validation_rows': 24,
        'source_train_sha256': sha256('data/manifests/pilot/train.jsonl'),
        'source_validation_sha256': sha256('data/manifests/pilot/validation.jsonl'),
        'created_utc': utc_now()})
    out = Path('configs/learning100')
    out.mkdir(exist_ok=True)
    for experiment in ('A0', 'A1', 'A3'):
        cfg = yaml.safe_load(Path(f'configs/{experiment}.yaml').read_text())
        cfg.update(train_manifest='data/manifests/learning100/train.jsonl',
                   validation_manifest='data/manifests/learning100/validation.jsonl',
                   style='data/style/learning100-train-mean.json',
                   labels='data/labels/learning100', steps=100, save_every=50)
        path = out / f'{experiment}.yaml'
        if path.exists() and yaml.safe_load(path.read_text()) != cfg:
            raise ResearchError('Existing learning configuration differs: ' + str(path))
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print('Fixed 100 training views, 24 validation views, and matched A0/A1/A3 configurations.')


if __name__ == '__main__':
    main()
