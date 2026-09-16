"""Run the fixed A1/A3 laptop comparison serially; persist status and fail on errors."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from satground.common import ROOT, ResearchError, load_json, provenance, save_json, sha256, utc_now


def run(output, report_output, resume=False):
    os.chdir(ROOT)
    root = Path(output)
    if root.exists() and any(root.iterdir()) and not resume:
        raise ResearchError('Use a fresh output directory or --resume.')
    root.mkdir(parents=True, exist_ok=True)
    manifest = 'data/manifests/learning100/validation.jsonl'
    manual_index = 'data/review/manual-scored-first3/approved-index.json'
    configs = {name: f'configs/exploratory500/{name}.yaml' for name in ('A1', 'A3')}
    tracked = [*configs.values(), manifest, 'data/manifests/learning100/train.jsonl', manual_index,
               'scripts/run_exploratory_pair.py', 'scripts/report_exploratory_pair.py']
    identity = dict(files={p: sha256(p) for p in tracked}, pipeline_source_hash=provenance()['pipeline_source_hash'])
    status_path = root / 'status.json'
    if resume:
        state = load_json(status_path)
        if state['identity'] != identity:
            raise ResearchError('Cannot resume after experiment inputs/code changed.')
    else:
        state = dict(created_utc=utc_now(), identity=identity, completed=[], status='starting',
                     study='exploratory_pending_data_review', report_output=str(report_output))
    def status(**updates):
        state.update(updates, updated_utc=utc_now())
        temporary = status_path.with_suffix('.tmp')
        save_json(temporary, state)
        os.replace(temporary, status_path)
    # One orchestrator per experiment directory. A crash leaves evidence, not a second GPU job.
    lock = root / 'runner.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        for name in ('A1', 'A3'):
            common = [sys.executable, '-u', '-m', 'satground.cli']
            train = [*common, 'train', '--config', configs[name], '--data-root', 'data/vigor', '--output', str(root / f'{name}-train')]
            if resume and (root / f'{name}-train/last.pt').exists():
                summary = root / f'{name}-train/summary.json'
                if summary.exists() and load_json(summary)['status'] == 'budget_complete':
                    if f'{name}-train' not in state['completed']:
                        state['completed'].append(f'{name}-train')
                else:
                    train.append('--resume')
            generate = [*common, 'generate', '--config', configs[name], '--manifest', manifest,
                        '--data-root', 'data/vigor', '--checkpoint', str(root / f'{name}-train/last.pt'),
                        '--output', str(root / f'{name}-validation')]
            evaluate = [*common, 'evaluate', '--manifest', manifest, '--predictions', str(root / f'{name}-validation'),
                        '--data-root', 'data/vigor', '--output', str(root / f'{name}-validation-eval'),
                        '--manual-index', manual_index, '--revision', 'ec86afeba68e656629ccf47e0c8d2902f964917b']
            for stage, command in ((f'{name}-train', train), (f'{name}-validation', generate), (f'{name}-validation-eval', evaluate)):
                if stage in state['completed']:
                    continue
                if provenance()['pipeline_source_hash'] != identity['pipeline_source_hash'] or any(sha256(p) != d for p, d in identity['files'].items()):
                    raise ResearchError('Experiment inputs/code changed during the run.')
                if stage.endswith('-eval') and (root / stage).exists():
                    raise ResearchError('Partial evaluation retained; choose a fresh output instead of overwriting.')
                status(status='running', current_stage=stage)
                print(f'{utc_now()} Starting {stage}', flush=True)
                with (root / f'{stage}.log').open('a', encoding='utf-8') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state['completed'].append(stage)
                status()
        status(status='reporting', current_stage='comparison')
        from report_exploratory_pair import report
        report(root, report_output)
        status(status='complete', current_stage=None)
        print('Complete:', report_output, flush=True)
    except BaseException as exc:
        status(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/exploratory500-seed17')
    parser.add_argument('--report-output', default='reports/exploratory500')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, args.report_output, args.resume)
