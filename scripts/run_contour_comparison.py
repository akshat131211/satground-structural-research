"""Run the fixed, fresh 500-step A2/corrected-A3 laptop comparison serially."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from satground.common import ROOT, ResearchError, load_json, provenance, save_json, sha256, utc_now

MANUAL='data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json'
MANIFEST='data/manifests/learning100/validation.jsonl'


def run(output, resume=False):
    os.chdir(ROOT);root=Path(output)
    if root.exists() and not resume:raise ResearchError('Use fresh output or compatible --resume.')
    support=load_json('reports/contour500/target-support.json')
    if not support['support_feasibility_passed']:raise ResearchError('Contour support preflight failed.')
    files=[MANUAL,MANIFEST,'data/manifests/learning100/train.jsonl','data/style/learning100-train-mean.json',
        'data/labels/contour100-v1/provenance.json','reports/contour500/target-support.json',
        'docs/CONTOUR_CORRECTION.md','scripts/run_contour_comparison.py','scripts/report_contour_comparison.py',
        'scripts/report_checkpoint_diagnosis.py','scripts/report_reviewed21.py','scripts/audit_exploratory500.py',
        'configs/contour500/A2.yaml','configs/contour500/A3.yaml']
    for entry in load_json(MANUAL).values():
        for key in ('target_image','building_mask','valid_mask'):
            if sha256(entry[key])!=entry[key+'_sha256']:raise ResearchError('Accepted mask/target changed.')
            files.append(entry[key])
    for sid,digest in load_json('data/labels/contour100-v1/provenance.json')['label_sha256'].items():
        path='data/labels/contour100-v1/'+sid+'.npz'
        if sha256(path)!=digest:raise ResearchError('Training contour label changed.')
        files.append(path)
    identity=dict(files={p:sha256(p) for p in files},pipeline_source_hash=provenance()['pipeline_source_hash'])
    root.mkdir(parents=True,exist_ok=True);state_path=root/'status.json'
    state=load_json(state_path) if resume else dict(created_utc=utc_now(),identity=identity,
        completed=[],study='exploratory_contour_support_correction',budget_steps_per_model=500,seed=17)
    if state['identity']!=identity:raise ResearchError('Pinned code or inputs differ; cannot resume.')

    def unchanged():
        if provenance()['pipeline_source_hash']!=identity['pipeline_source_hash'] or any(
            sha256(p)!=h for p,h in identity['files'].items()):raise ResearchError('Pinned source/input changed.')

    def status(**updates):
        state.update(updates,updated_utc=utc_now());save_json(state_path,state)

    lock=root/'runner.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        for name in ('A2','A3'):
            cfg=f'configs/contour500/{name}.yaml'
            base=[sys.executable,'-u','-m','satground.cli']
            train=[*base,'train','--config',cfg,'--data-root','data/vigor','--output',str(root/f'{name}-train')]
            if resume and (root/f'{name}-train/last.pt').exists():
                summary=root/f'{name}-train/summary.json'
                if summary.exists() and load_json(summary)['status']=='budget_complete':
                    if name+'-train' not in state['completed']:state['completed'].append(name+'-train')
                else:train.append('--resume')
            generate=[*base,'generate','--config',cfg,'--manifest',MANIFEST,'--data-root','data/vigor',
                '--checkpoint',str(root/f'{name}-train/last.pt'),'--output',str(root/f'{name}-validation')]
            evaluate=[*base,'evaluate','--manifest',MANIFEST,'--data-root','data/vigor',
                '--predictions',str(root/f'{name}-validation'),'--output',str(root/f'{name}-validation-eval'),
                '--manual-index',MANUAL,'--revision','ec86afeba68e656629ccf47e0c8d2902f964917b']
            for stage,command in [(name+'-train',train),(name+'-validation',generate),(name+'-validation-eval',evaluate)]:
                if stage in state['completed']:continue
                unchanged()
                if not stage.endswith('-train') and (root/stage).exists():
                    raise ResearchError('Partial prediction/evaluation retained; inspect before resuming.')
                status(status='running',current_stage=stage)
                print(utc_now(),stage,flush=True)
                with (root/(stage+'.log')).open('a',encoding='utf-8') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                state['completed'].append(stage);status()
        unchanged();status(status='reporting',current_stage='report')
        from report_contour_comparison import report
        report(root,Path('reports/contour500/comparison'),root/'gallery')
        status(status='complete',current_stage=None)
    except BaseException as error:
        status(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/contour500-seed17')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.output,a.resume)
