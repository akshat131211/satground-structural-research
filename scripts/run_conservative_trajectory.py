"""Evaluate all conservative checkpoints with completed human labels; never train."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from satground.common import ROOT, ResearchError, load_json, provenance, save_json, sha256, utc_now


def run(output, resume=False):
    os.chdir(ROOT)
    root = Path(output)
    manifest = 'data/manifests/learning100/validation.jsonl'
    manual = 'data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json'
    files = [manifest, manual, 'data/style/learning100-train-mean.json',
             'docs/LONG_TRAINING_ASSESSMENT.md', 'scripts/run_conservative_trajectory.py',
             'data/manifests/learning100/train.jsonl']
    for item in load_json(manual).values():
        for key in ('target_image', 'building_mask', 'valid_mask'):
            if sha256(item[key]) != item[key + '_sha256']:
                raise ResearchError('An accepted review artifact changed.')
            files.append(item[key])
    for name in ('A1', 'A2', 'A3'):
        files.append(f'configs/conservative500/{name}.yaml')
        files.extend(f'runs/conservative500-seed17/{name}-train/step-{step:06d}.pt'
                     for step in (100, 200, 300, 400, 500))
    identity = {'files': {p: sha256(p) for p in files},
                'pipeline_source_hash': provenance()['pipeline_source_hash']}
    if root.exists() and not resume:
        raise ResearchError('Use fresh output or --resume; prior outputs are preserved.')
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'status.json'
    state = load_json(state_path) if resume else dict(created_utc=utc_now(), identity=identity, completed=[], new_training_performed=False)
    if state['identity'] != identity:
        raise ResearchError('Checkpoint diagnosis inputs changed.')

    def status(**updates):
        state.update(updates, updated_utc=utc_now())
        save_json(state_path.with_suffix('.tmp'), state)
        os.replace(state_path.with_suffix('.tmp'), state_path)

    lock = root / 'runner.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        for step in (100, 200, 300, 400, 500):
            for name in ('A1', 'A2', 'A3'):
                prefix = f'{name}-{step:06d}'
                base = [sys.executable, '-u', '-m', 'satground.cli']
                generate = [*base, 'generate', '--config', f'configs/conservative500/{name}.yaml',
                            '--manifest', manifest, '--data-root', 'data/vigor',
                            '--checkpoint', f'runs/conservative500-seed17/{name}-train/step-{step:06d}.pt',
                            '--output', str(root / prefix)]
                evaluate = [*base, 'evaluate', '--manifest', manifest, '--data-root', 'data/vigor',
                            '--predictions', str(root / prefix), '--output', str(root / (prefix + '-eval')),
                            '--manual-index', manual, '--revision', 'ec86afeba68e656629ccf47e0c8d2902f964917b']
                for stage, command, destination in ((prefix + '-generate', generate, root / prefix),
                                                     (prefix + '-evaluate', evaluate, root / (prefix + '-eval'))):
                    if stage in state['completed']:
                        continue
                    if destination.exists():
                        raise ResearchError('Partial output retained; use a fresh diagnosis root.')
                    if provenance()['pipeline_source_hash'] != identity['pipeline_source_hash'] or any(
                            sha256(p) != digest for p, digest in identity['files'].items()):
                        raise ResearchError('Code or inputs changed during diagnosis.')
                    status(status='running', current_stage=stage)
                    print(utc_now(), stage, flush=True)
                    with (root / (stage + '.log')).open('w', encoding='utf-8') as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
                    state['completed'].append(stage)
                    status()
        status(status='complete', current_stage=None)
    except BaseException as exc:
        status(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/conservative-trajectory-reviewed21')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, args.resume)
