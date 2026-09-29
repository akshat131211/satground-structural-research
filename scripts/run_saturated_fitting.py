"""One serial fresh A0/A2/saturated-A3 pair, including its one-update startup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import psutil
import torch

from prepare_saturated_fitting import OUTPUT as TRIAL, source_pins, verify_trial
from prepare_reviewed_fitting import check_pins, require
from probe_reviewed_contour_gradients import LOCK as HISTORICAL_LOCK
from probe_surface_candidate_gradients import LOCK as V1_LOCK
from probe_saturated_surface_gradients import LOCK as V2_LOCK
from run_reviewed_fitting import LOCK as ORIGINAL_LOCK, other_gpu_work
from satground.common import ROOT, load_json, object_hash, provenance, save_json, sha256, utc_now
from train_saturated_fitting import fitting_identity


LOCK = ROOT / 'runs/saturated-fitting-gpu.lock'
OUTPUT = ROOT / 'runs/saturated-fitting200-seed17'


def require_runner():
    token = os.environ.get('SATGROUND_SATURATED_RUNNER_TOKEN')
    require(bool(token) and LOCK.is_file(), 'Launch saturated fitting through its serial runner.')
    lock = load_json(LOCK)
    require(lock.get('token') == token, 'Saturated runner lock mismatch.')
    parents = {p.pid: p for p in psutil.Process().parents()}
    require(lock['pid'] in parents and parents[lock['pid']].create_time() == lock['create_time'],
            'The saturated runner no longer owns this child.')


def make_stages(trial, root, data_root):
    trial, root = Path(trial).resolve(), Path(root).resolve()
    record = load_json(trial / 'trial.json')
    original = Path(record['original_bundle'])
    manifest, manual = str(original / 'manifest.jsonl'), str(original / 'manual-index.json')
    base = [sys.executable, '-u', '-m', 'satground.cli']
    result = []
    for name in ('A0', 'A2', 'A3'):
        cfg = str(trial / f'{name}.yaml')
        if name != 'A0':
            result.append((name + '-train', [sys.executable, '-u', str(ROOT / 'scripts/train_saturated_fitting.py'),
                '--config', cfg, '--data-root', str(data_root), '--output', str(root / f'{name}-train')]))
        generation = [*base, 'generate', '--config', cfg, '--manifest', manifest, '--data-root', str(data_root),
                      '--output', str(root / f'{name}-fitting')]
        if name != 'A0':
            generation += ['--checkpoint', str(root / f'{name}-train/last.pt')]
        result.append((name + '-fitting', generation))
        result.append((name + '-eval', [*base, 'evaluate', '--manifest', manifest, '--data-root', str(data_root),
            '--predictions', str(root / f'{name}-fitting'), '--output', str(root / f'{name}-eval'),
            '--manual-index', manual, '--revision', record['protocol']['evaluation_segmenter_revision']]))
    return result


def run(trial=TRIAL, output=OUTPUT, data_root='data/vigor', resume=False, startup_one_update=False):
    os.chdir(ROOT)
    trial, root = Path(trial).resolve(), Path(output).resolve()
    require(root == OUTPUT.resolve() and not (resume and startup_one_update), 'Use the fixed private saturated run and one startup mode.')
    require(resume or startup_one_update, 'A fresh saturated trial requires --startup-one-update within the fixed budget.')
    record = verify_trial(trial)
    original = Path(record['original_bundle'])
    require(resume or not root.exists(), 'Use the sole fresh run or a verified compatible resume.')
    identities = {n: fitting_identity(trial / f'{n}.yaml', data_root)[2] for n in ('A2', 'A3')}
    identity = dict(trial=str(trial), bundle=str(original), bundle_sha256=record['original_bundle_sha256'],
        files={**record['source_file_sha256'], **record['artifact_sha256'], **source_pins(),
               str(trial / 'trial.json'): sha256(trial / 'trial.json')},
        training_identity={n: object_hash(value) for n, value in identities.items()},
        pipeline_source_hash=record['pipeline_source_hash'])
    state = load_json(root / 'status.json') if resume else dict(created_utc=utc_now(), identity=identity,
        completed=[], scope='training_fitting_only', budget_steps_per_model=200, seed=17,
        server_gate_eligible=False, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip())
    require(state['identity'] == identity, 'Saturated runner inputs/source differ; preserve outputs and diagnose.')
    if not resume:
        require(not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
                'Commit checked saturated source before launching the recorded trial.')
    require(not any(p.exists() for p in (LOCK, ORIGINAL_LOCK, HISTORICAL_LOCK, V1_LOCK, V2_LOCK))
            and not other_gpu_work(), 'A project lock or Python/GPU worker exists; do not launch duplicate work.')
    root.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    with LOCK.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(), token=token, output=str(root)), stream)
    child = None

    def status(**updates):
        state.update(updates, updated_utc=utc_now())
        save_json(root / 'status.json', state)

    def unchanged():
        check_pins(identity['files'])
        require(provenance()['pipeline_source_hash'] == identity['pipeline_source_hash'], 'Core source changed.')

    try:
        stages = make_stages(trial, root, data_root)
        completed = state['completed']
        require(completed == [name for name, _ in stages[:len(completed)]], 'Completed stages are not a serial prefix.')
        for stage, command in stages:
            if stage in completed:
                continue
            unchanged()
            target = root / stage
            restore = None
            if target.exists():
                require(resume and stage.endswith('-train') and (target / 'last.pt').is_file(),
                        'Partial stage outputs retained; inspect before recovery.')
                checkpoint = torch.load(target / 'last.pt', map_location='cpu', weights_only=True)
                model = stage.split('-')[0]
                require(checkpoint['identity_hash'] == identity['training_identity'][model]
                        and type(checkpoint['step']) is int and 0 < checkpoint['step'] <= 200,
                        'Incompatible checkpoint; no recovery attempted.')
                restore = dict(checkpoint_step=checkpoint['step'], remaining_updates=200 - checkpoint['step'],
                               exact_prefix_and_rng_verification_required=True)
                command += ['--resume']
            elif startup_one_update and stage == 'A2-train':
                command += ['--startup-one-update']
            status(status='running', current_stage=stage, command=command, child_pid=None,
                   pending_resume_verification=restore)
            env = dict(os.environ, SATGROUND_SATURATED_RUNNER_TOKEN=token)
            with (root / (stage + '.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=env)
                status(child_pid=child.pid)
                require(child.wait() == 0, f'{stage} failed; preserve its log and outputs.')
            if startup_one_update and stage == 'A2-train':
                receipt = load_json(target / 'startup-receipt.json')
                require(receipt['status'] == 'startup_complete' and receipt['step'] == 1
                        and receipt['identity_hash'] == identity['training_identity']['A2']
                        and receipt['checkpoint_sha256'] == sha256(target / 'last.pt')
                        and receipt['startup_checkpoint_sha256'] == receipt['checkpoint_sha256'] == sha256(target / 'startup.pt')
                        and receipt['updates_included_in_budget'] is True and receipt['remaining_updates'] == 199,
                        'Startup checkpoint receipt is invalid.')
                status(status='startup_complete', child_pid=None, startup_update=1,
                       startup_checkpoint_sha256=receipt['checkpoint_sha256'],
                       startup_prefix_verified=receipt['sample_prefix_verified'], remaining_A2_updates=199)
                return state
            if restore:
                proof = load_json(target / 'resume-verification.json')
                require(proof['identity_hash'] == identity['training_identity'][stage.split('-')[0]]
                        and proof['restored_step'] == restore['checkpoint_step']
                        and proof['remaining_updates'] == restore['remaining_updates']
                        and proof['sample_prefix_verified'] is True and proof['checkpoint_payload_verified'] is True
                        and (proof['adapter_optimizer_scaler_rng_restored'] is True
                             and proof['exact_adapter_optimizer_scaler_readback_verified'] is True
                             and proof['exact_python_numpy_torch_cpu_cuda_rng_readback_verified'] is True if restore['checkpoint_step'] < 200
                             else proof['completion_without_optimizer_updates'] is True
                                  and proof['adapter_optimizer_scaler_rng_restored'] is False
                                  and proof['optimizer_updates_performed'] == 0), 'Resume verification is incomplete.')
                state.setdefault('verified_resumes', {})[stage] = proof
            completed.append(stage)
            status(child_pid=None, pending_resume_verification=None)
        unchanged()
        status(status='reporting', current_stage='report', child_pid=None)
        from report_saturated_fitting import report
        report(root, root / 'report')
        status(status='complete', current_stage=None, child_pid=None)
        return state
    except BaseException as error:
        status(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if child is None or child.poll() is not None:
            if LOCK.exists() and load_json(LOCK).get('token') == token:
                LOCK.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', default=str(TRIAL))
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--data-root', default='data/vigor')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--startup-one-update', action='store_true')
    args = parser.parse_args()
    run(args.trial, args.output, args.data_root, args.resume, args.startup_one_update)
