"""Re-score saved A0-A3 images under the completed 21-target label index.

Run from the checkout root with locally available data and approved labels.
Never trains, regenerates images, or overwrites a previous evaluation directory.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from satground.common import ResearchError, load_json, provenance, save_json, sha256, utc_now


def evaluate(root, index):
    root, index = Path(root), Path(index)
    entries = load_json(index)
    if len(entries) != 21:
        raise ResearchError('Expected original three plus eighteen approved targets.')
    manifest = Path('data/manifests/learning100/validation.jsonl')
    folders = {'A0': 'runs/exploratory500-A0-reference', **{
        n: f'runs/conservative500-seed17/{n}-validation' for n in ('A1', 'A2', 'A3')}}
    files = [index, manifest, Path('data/review/manual-scored-first3/approved-index.json')]
    for entry in entries.values():
        files += [Path(entry[k]) for k in ('target_image', 'building_mask', 'valid_mask')]
    for folder in folders.values():
        files += [Path(folder) / 'predictions.jsonl', Path(folder) / 'generation.json',
                  Path(folder + '-eval') / 'metrics.jsonl', Path(folder + '-eval') / 'summary.json']
    state = dict(created_utc=utc_now(), status='running', completed=[], current_model=None,
                 identity=dict(files={str(p): sha256(p) for p in files},
                               pipeline_source_hash=provenance()['pipeline_source_hash']),
                 model_predictions_reused=True, new_training_performed=False, reviewed_targets=21)
    root.mkdir(parents=True, exist_ok=False)

    def update(**kwargs):
        state.update(kwargs, updated_utc=utc_now())
        save_json(root / 'status.json', state)

    def verify_inputs():
        if not all(sha256(p) == h for p, h in state['identity']['files'].items()):
            raise ResearchError('Re-scoring inputs changed; partial outputs retained.')
        if provenance()['pipeline_source_hash'] != state['identity']['pipeline_source_hash']:
            raise ResearchError('Evaluation source changed; partial outputs retained.')

    try:
        for name, folder in folders.items():
            verify_inputs()
            update(current_model=name)
            print(utc_now(), name, 'evaluate', flush=True)
            with (root / (name + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-m', 'satground', 'evaluate',
                    '--manifest', str(manifest), '--predictions', folder, '--data-root', 'data/vigor',
                    '--output', str(root / (name + '-eval')), '--manual-index', str(index),
                    '--revision', 'ec86afeba68e656629ccf47e0c8d2902f964917b'],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            verify_inputs()
            state['completed'].append(name)
        update(status='complete', current_model=None)
    except BaseException as error:
        update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Fresh local output directory.')
    parser.add_argument('--index', default='data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json')
    args = parser.parse_args()
    evaluate(args.output, args.index)
