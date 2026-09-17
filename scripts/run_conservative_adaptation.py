"""Run the single diagnosed learning-rate revision with matched A1/A2/A3 budgets."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from satground.common import ROOT, ResearchError, load_json, provenance, save_json, sha256, utc_now


def run(root, report_output, resume=False):
    os.chdir(ROOT)
    root = Path(root)
    manifest = 'data/manifests/learning100/validation.jsonl'
    manual = 'data/review/manual-scored-first3/approved-index.json'
    files = [manifest, manual, 'data/manifests/learning100/train.jsonl',
             'data/style/learning100-train-mean.json', 'docs/CONSERVATIVE_ADAPTATION.md',
             'scripts/run_conservative_adaptation.py', 'scripts/report_conservative_adaptation.py',
             'scripts/report_checkpoint_diagnosis.py', 'scripts/report_exploratory_pair.py',
             'scripts/audit_exploratory500.py']
    files += [f'configs/conservative500/{n}.yaml' for n in ('A1', 'A2', 'A3')]
    for item in load_json(manual).values():
        for key in ('target_image', 'building_mask', 'valid_mask'):
            if sha256(item[key]) != item[key + '_sha256']:
                raise ResearchError('Accepted review artifact changed.')
            files.append(item[key])
    identity = dict(files={p: sha256(p) for p in files}, pipeline_source_hash=provenance()['pipeline_source_hash'])
    if root.exists() and not resume:
        raise ResearchError('Use fresh output or compatible --resume.')
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'status.json'
    state = load_json(state_path) if resume else dict(created_utc=utc_now(), identity=identity, completed=[],
                                                      report_output=str(report_output), study='exploratory_pending_data_review')
    if state['identity'] != identity:
        raise ResearchError('Code or inputs changed since this experiment started.')

    def status(**updates):
        state.update(updates, updated_utc=utc_now())
        save_json(state_path.with_suffix('.tmp'), state)
        os.replace(state_path.with_suffix('.tmp'), state_path)

    lock = root / 'runner.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        for name in ('A1', 'A2', 'A3'):
            base = [sys.executable, '-u', '-m', 'satground.cli']
            config = f'configs/conservative500/{name}.yaml'
            train = [*base, 'train', '--config', config, '--data-root', 'data/vigor', '--output', str(root / f'{name}-train')]
            if resume and (root / f'{name}-train/last.pt').exists():
                train_summary = root / f'{name}-train/summary.json'
                if train_summary.exists() and load_json(train_summary)['status'] == 'budget_complete':
                    if name + '-train' not in state['completed']:
                        state['completed'].append(name + '-train')
                else:
                    train.append('--resume')
            generate = [*base, 'generate', '--config', config, '--manifest', manifest, '--data-root', 'data/vigor',
                        '--checkpoint', str(root / f'{name}-train/last.pt'), '--output', str(root / f'{name}-validation')]
            evaluate = [*base, 'evaluate', '--manifest', manifest, '--predictions', str(root / f'{name}-validation'),
                        '--data-root', 'data/vigor', '--output', str(root / f'{name}-validation-eval'),
                        '--manual-index', manual, '--revision', 'ec86afeba68e656629ccf47e0c8d2902f964917b']
            for stage, command in ((name + '-train', train), (name + '-validation', generate), (name + '-validation-eval', evaluate)):
                if stage in state['completed']:
                    continue
                if provenance()['pipeline_source_hash'] != identity['pipeline_source_hash'] or any(
                        sha256(p) != d for p, d in identity['files'].items()):
                    raise ResearchError('Pinned source or inputs changed during run.')
                if stage.endswith('-eval') and (root / stage).exists():
                    raise ResearchError('Partial evaluation retained; use a fresh output.')
                status(status='running', current_stage=stage)
                print(utc_now(), stage, flush=True)
                with (root / (stage + '.log')).open('a', encoding='utf-8') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                state['completed'].append(stage)
                status()
        status(status='reporting', current_stage='report')
        from report_conservative_adaptation import report
        report(root, report_output)
        status(status='complete', current_stage=None)
    except BaseException as exc:
        status(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/conservative500-seed17')
    parser.add_argument('--report-output', default='reports/conservative500')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, args.report_output, args.resume)
