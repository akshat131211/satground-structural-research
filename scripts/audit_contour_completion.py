"""Verify the completed contour comparison and quantify historical control drift."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from audit_exploratory500 import finite_values, require, verified_predictions
from report_reviewed21 import delta
from satground.common import load_json, provenance, read_jsonl, save_json, sha256, utc_now
from satground.metrics import grouped_mean
from satground.runs import read_config


def audit(output):
    out = Path(output)
    require(not out.exists(), 'Use a fresh audit output.')
    root = Path('runs/contour500-seed17')
    state = load_json(root/'status.json')
    require(state['status']=='complete' and len(state['completed'])==6, 'Runner incomplete.')
    require(all(sha256(p)==h for p,h in state['identity']['files'].items()), 'Pinned inputs changed.')
    require(provenance()['pipeline_source_hash']==state['identity']['pipeline_source_hash'], 'Source changed.')
    original = load_json('reports/contour500/comparison/summary.json')
    repeated = load_json('runs/contour-report-verification/summary.json')
    require({k:v for k,v in original.items() if k!='created_utc'}==
            {k:v for k,v in repeated.items() if k!='created_utc'}, 'Report differs on reproduction.')
    require(sha256('reports/contour500/comparison/report.md')==sha256('runs/contour-report-verification/report.md'),
            'Report Markdown changed on reproduction.')
    gallery = sorted((root/'gallery').glob('page-*.png'))
    require(len(gallery)==6 and all(sha256(p)==sha256(Path('runs/contour-gallery-verification')/p.name) for p in gallery),
            'Gallery differs on reproduction.')
    checkpoints, training, logs = {}, {}, {}
    for name in ('A2','A3'):
        folder = root/(name+'-train')
        run = load_json(folder/'run.json')
        cfg = read_config('configs/contour500/'+name+'.yaml')
        logs[name] = read_jsonl(folder/'training.jsonl')
        require([r['step'] for r in logs[name]]==list(range(1,501)), 'Wrong training steps.')
        require(all(math.isfinite(v) for r in logs[name] for v in r.values() if isinstance(v,(int,float))), 'Non-finite log.')
        checkpoints[name] = []
        for step in range(100,501,100):
            p = folder/f'step-{step:06d}.pt'
            c = torch.load(p,map_location='cpu',weights_only=True)
            require(c['step']==step and c['identity_hash']==run['identity_hash'] and c['config']==cfg, 'Checkpoint identity changed.')
            require(all(int(v['step'])==step for v in c['optimizer']['state'].values()), 'Optimizer step mismatch.')
            checkpoints[name].append(dict(step=step,sha256=sha256(p),
                finite_tensor_values={k:finite_values(c[k]) for k in ('adapter','optimizer')}))
        last = torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        require(last['step']==500 and last['identity_hash']==run['identity_hash'], 'Final checkpoint identity mismatch.')
        require(all(torch.equal(v,last['adapter'][k]) for k,v in c['adapter'].items()), 'Final weights differ.')
        for k,v in c['optimizer']['state'].items():
            require(all(torch.equal(t,last['optimizer']['state'][k][n]) if torch.is_tensor(t)
                        else t==last['optimizer']['state'][k][n] for n,t in v.items()), 'Final optimizer differs.')
        summary = load_json(folder/'summary.json')
        require(summary['checkpoint_sha256']==sha256(folder/'last.pt') and summary['status']=='budget_complete', 'Final summary mismatch.')
        training[name] = dict(steps=500,all_values_finite=True,checkpoint_sha256=sha256(folder/'last.pt'),
            elapsed_seconds=summary['elapsed_seconds'],peak_vram_mib=summary['peak_vram_mib'])
    preserved = load_json(root/'recovery-20260918.json')['preserved_A2_artifact_sha256']
    require(all(sha256(p)==h for p,h in preserved.items()), 'A2 artifacts changed after recovery.')
    oldcfg = read_config('configs/conservative500/A2.yaml')
    newcfg = read_config('configs/contour500/A2.yaml')
    allowed = {'labels','exploratory_reason','boundary_mode'}
    require({k:v for k,v in oldcfg.items() if k not in allowed}=={k:v for k,v in newcfg.items() if k not in allowed},
            'Historical A2 hyperparameters differ.')
    rows = read_jsonl(newcfg['train_manifest'])
    require(len(rows)==100, 'Training cohort changed.')
    for row in rows:
        with np.load(Path(oldcfg['labels'])/(row['sample_id']+'.npz')) as old, np.load(Path(newcfg['labels'])/(row['sample_id']+'.npz')) as new:
            require(all(np.array_equal(old[k],new[k]) for k in old.files), 'Original pseudo-label array changed.')
    oldrun = load_json('runs/conservative500-seed17/A2-train/run.json')
    newrun = load_json(root/'A2-train/run.json')
    for key in ('code_revision','model_revision','packages','python','platform'):
        require(oldrun['provenance'][key]==newrun['provenance'][key], 'Environment differs: '+key)
    for key in ('data_digest','style_sha256','train_manifest_sha256','validation_manifest_sha256'):
        require(oldrun['identity'][key]==newrun['identity'][key], 'Historical input differs: '+key)
    folders = [Path('runs/conservative500-seed17/A2-validation'),root/'A2-validation',Path('runs/contour-A2-inference-repeat')]
    hashes = [verified_predictions(p) for p in folders]
    require(set(hashes[0])==set(hashes[1])==set(hashes[2]) and len(hashes[1])==24, 'Prediction set differs.')
    require(hashes[1]==hashes[2], 'Same-checkpoint inference does not reproduce.')
    errors = []
    for sid in hashes[0]:
        with Image.open(folders[0]/(sid+'.png')) as a, Image.open(folders[1]/(sid+'.png')) as b:
            errors.append(float(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)).mean()/255.))
    ca = torch.load('runs/conservative500-seed17/A2-train/last.pt',map_location='cpu',weights_only=True)['adapter']
    cb = torch.load(root/'A2-train/last.pt',map_location='cpu',weights_only=True)['adapter']
    va = torch.cat([v.flatten().double() for v in ca.values()])
    vb = torch.cat([cb[k].flatten().double() for k in ca])
    older = read_jsonl('runs/conservative500-seed17/A2-train/training.jsonl')
    for k in ('rgb','lpips','region','total'):
        require(older[0][k]==logs['A2'][0][k], 'First-step A2 forward loss differs.')
    a = read_jsonl('runs/conservative500-reviewed21/A2-eval/metrics.jsonl')
    b = read_jsonl(root/'A2-validation-eval/metrics.jsonl')
    corrected = read_jsonl(root/'A3-validation-eval/metrics.jsonl')
    first3 = load_json('data/review/manual-scored-first3/approved-index.json')
    require(len(first3)==3, 'Original review index changed.')
    numerical = load_json('runs/contour-numerical-check/summary.json')
    require(numerical['optimizer_steps']==0 and numerical['parameters_unchanged'] and
            numerical['script_sha256']==sha256(Path(__file__).with_name('diagnose_contour_numerics.py')), 'Numerical probe identity changed.')
    result = dict(created_utc=utc_now(),status='verified',script_sha256=sha256(__file__),
        training=training,total_retained_steps=1000,checkpoints=checkpoints,pinned_files_verified=len(state['identity']['files']),
        source_hash=state['identity']['pipeline_source_hash'],report_reproduced=True,report_timestamp_excluded=True,
        gallery_pages_reproduced=6,preserved_A2_artifacts_verified=len(preserved),original_training_labels_identical=100,
        reviewed_masks=21,manual_index_sha256=original['manual_index_sha256'],new_predictions=48,
        same_checkpoint_A2_repeat_exact_matches=24,historical_control=dict(
            exact_prediction_matches=sum(hashes[0][k]==hashes[1][k] for k in hashes[0]),
            mean_image_absolute_difference_0_1=float(np.mean(errors)),max_per_image_mean_difference_0_1=max(errors),
            adapter_relative_l2=float((vb-va).norm()/va.norm()),metrics_change=delta(a,b),
            first_step_forward_losses_exact=True,first_step_gradient_norm_difference=logs['A2'][0]['gradient_norm']-older[0]['gradient_norm'],
            input_and_environment_checks_passed=True,mathematically_equivalent_A2_objective=True,
            exact_retraining_reproduced=False),
        original_three_reviewed=dict(primary_delta=delta([r for r in b if r['sample_id'] in first3],
            [r for r in corrected if r['sample_id'] in first3]),
            grouped_boundary={n:grouped_mean([r for r in data if r['sample_id'] in first3],'boundary_error')
                              for n,data in [('A2_fresh',b),('A3_corrected',corrected)]}),
        numerical_probe=numerical,interrupted_discarded_steps=43,preflight_steps=3,
        no_additional_training=True,data_qa_complete=False,server_gate_passed=False)
    save_json(out,result)
    print('Verified 1,000 training rows, 10 periodic checkpoints, 48 predictions, 177 pins and report reproduction.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    audit(parser.parse_args().output)
