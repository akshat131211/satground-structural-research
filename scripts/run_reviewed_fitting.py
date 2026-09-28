"""One serial frozen-reference/A2/A3 fitting diagnosis; no open-ended search."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import psutil

from prepare_reviewed_fitting import check_pins, require, verify_bundle
from satground.common import ROOT, load_json, object_hash, provenance, save_json, sha256, utc_now
from train_reviewed_fitting import fitting_identity, source_pins

LOCK = ROOT / 'runs/reviewed-fitting-gpu.lock'


def other_gpu_work():
    """Fail closed on another project Python worker or a reported CUDA process."""
    own = psutil.Process()
    allowed_pids = {own.pid, *[p.pid for p in own.parents()]}
    found = set()
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid in allowed_pids or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            command = ' '.join(process.info['cmdline'] or [])
            if any(script in command for script in ('open_training_review.py', 'open_additional_review.py')):
                continue
            if Path(process.cwd()).resolve() == ROOT.resolve():
                found.add(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            raise RuntimeError('Cannot inspect another Python process; verify its GPU use before launch.')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    for line in result.stdout.splitlines():
        if line.strip().isdigit() and int(line) not in allowed_pids:
            found.add(int(line))
    return sorted(found)


def require_runner():
    token = os.environ.get('SATGROUND_FITTING_RUNNER_TOKEN')
    require(bool(token) and LOCK.is_file(), 'Launch fitting through the serial runner and its GPU lock.')
    lock = load_json(LOCK)
    require(lock.get('token') == token, 'Fitting runner lock mismatch.')
    parents = {p.pid: p for p in psutil.Process().parents()}
    require(lock['pid'] in parents and parents[lock['pid']].create_time() == lock['create_time'],
            'Fitting runner no longer owns this child.')


def make_stages(bundle, root, data_root):
    bundle, root = Path(bundle), Path(root)
    base = [sys.executable, '-u', '-m', 'satground.cli']
    result = []
    for name in ('A0', 'A2', 'A3'):
        cfg = str(bundle / f'{name}.yaml')
        checkpoint = root / f'{name}-train/last.pt'
        if name != 'A0':
            result.append((name + '-train', [sys.executable, '-u', str(ROOT / 'scripts/train_reviewed_fitting.py'),
                '--config', cfg, '--data-root', str(data_root), '--output', str(root / f'{name}-train')]))
        generate = [*base, 'generate', '--config', cfg, '--manifest', str(bundle / 'manifest.jsonl'),
                    '--data-root', str(data_root), '--output', str(root / f'{name}-fitting')]
        if name != 'A0':
            generate.extend(['--checkpoint', str(checkpoint)])
        result.append((name + '-fitting', generate))
        revision = load_json(bundle / 'bundle.json')['protocol']['evaluation_segmenter_revision']
        result.append((name + '-eval', [*base, 'evaluate', '--manifest', str(bundle / 'manifest.jsonl'),
            '--data-root', str(data_root), '--predictions', str(root / f'{name}-fitting'),
            '--output', str(root / f'{name}-eval'), '--manual-index', str(bundle / 'manual-index.json'), '--revision', revision]))
    return result


def run(bundle, output, data_root, resume=False):
    os.chdir(ROOT)
    bundle, root = Path(bundle).resolve(), Path(output).resolve()
    require(root.is_relative_to(ROOT / 'runs'), 'Keep all raw fitting outputs under ignored runs/.')
    record = verify_bundle(bundle)
    require(resume or not root.exists(), 'Use fresh output or a verified compatible resume.')
    identities = {n: fitting_identity(bundle / f'{n}.yaml', data_root)[2] for n in ('A2', 'A3')}
    identity = dict(bundle=str(bundle), bundle_sha256=sha256(bundle / 'bundle.json'),
                    files={**record['source_file_sha256'], **record['artifact_sha256'], **source_pins()},
                    pipeline_source_hash=provenance()['pipeline_source_hash'],
                    training_identity={n: object_hash(v) for n, v in identities.items()})
    state = load_json(root / 'status.json') if resume else dict(created_utc=utc_now(), identity=identity,
        completed=[], scope='training_fitting_only', budget_steps_per_model=200, seed=17, server_gate_eligible=False,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip())
    require(state['identity'] == identity, 'Runner inputs or source differ; preserve outputs and diagnose.')
    if not resume:
        require(not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
                'Commit the checked fitting implementation before launching this recorded experiment.')
    require(not LOCK.exists(), 'A fitting lock exists; inspect runner and child processes before any recovery.')
    require(not other_gpu_work(), 'Another Python/GPU job is present; do not launch duplicate GPU work.')
    root.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(exist_ok=True)
    token = uuid4().hex
    lock = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), token=token, output=str(root))
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(lock, stream)
    child = None

    def status(**updates):
        state.update(updates, updated_utc=utc_now())
        save_json(root / 'status.json', state)

    def unchanged():
        check_pins(identity['files'])
        require(sha256(bundle / 'bundle.json') == identity['bundle_sha256']
                and provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Pinned fitting evidence changed.')

    try:
        stages = make_stages(bundle, root, data_root)
        completed = state['completed']
        require(completed == [s[0] for s in stages[:len(completed)]], 'Completed stages are not a serial prefix.')
        for stage, command in stages:
            if stage in completed:
                continue
            unchanged()
            target = root / stage
            if target.exists():
                require(resume and stage.endswith('-train') and (target / 'last.pt').is_file(),
                        'Partial outputs preserved; inspect them before deciding recovery.')
                command = [*command, '--resume']
            status(status='running', current_stage=stage, command=command, child_pid=None)
            env = dict(os.environ, SATGROUND_FITTING_RUNNER_TOKEN=token)
            with (root / (stage + '.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=env)
                status(child_pid=child.pid)
                code = child.wait()
                require(code == 0, f'{stage} exited {code}; preserve its log and outputs.')
            completed.append(stage)
            status(child_pid=None)
        unchanged()
        status(status='reporting', current_stage='report')
        from report_reviewed_fitting import report
        report(root, root / 'report')
        status(status='complete', current_stage=None, child_pid=None)
    except BaseException as error:
        status(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        # If an interrupted wait left the child alive, keep the lock for inspection.
        if child is None or child.poll() is not None:
            if LOCK.exists() and load_json(LOCK).get('token') == token:
                LOCK.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', default='runs/reviewed-fitting200-seed17')
    parser.add_argument('--data-root', default='data/vigor')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.bundle, args.output, args.data_root, args.resume)
