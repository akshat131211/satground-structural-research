"""Verify and summarize the bounded contour correction without publishing samples."""
import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from audit_exploratory500 import finite_values, require, verified_predictions
from report_checkpoint_diagnosis import strata
from report_reviewed21 import delta
from satground.common import load_json, read_jsonl, sha256, utc_now
from satground.metrics import grouped_mean

METRICS=('boundary_error','building_iou','lpips','ssim')
MANUAL='data/review/checkpoint-review18-approved/combined-original3-additional18-index-v1.json'


def check_matched_configs(control,proposed):
    require(control['experiment']=='A2' and proposed['experiment']=='A3','Unexpected compared models.')
    require(control['boundary_weight']==0 and proposed['boundary_weight']==.5,'Unexpected boundary weights.')
    allowed={'experiment','boundary_weight'}
    require({k:v for k,v in control.items() if k not in allowed}==
            {k:v for k,v in proposed.items() if k not in allowed},'Unmatched training resources.')
    require(control['steps']==500 and control['seed']==17 and control['learning_rate']==.00003 and
            control['boundary_mode']=='anchored_contour_v1','Wrong frozen contour protocol.')


def report(root,output,local):
    root,output,local=map(Path,(root,output,local))
    require(not output.exists() and not local.exists(),'Use fresh report/gallery outputs.')
    state=load_json(root/'status.json')
    expected=[n+suffix for n in ('A2','A3') for suffix in ('-train','-validation','-validation-eval')]
    require(state['status'] in ('reporting','complete') and state['completed']==expected,'Comparison stages incomplete.')
    require(all(sha256(p)==h for p,h in state['identity']['files'].items()),'Pinned input/source changed.')
    index=load_json(MANUAL);require(len(index)==21,'Reviewed label count changed.')
    for item in index.values():
        for k in ('target_image','building_mask','valid_mask'):
            require(sha256(item[k])==item[k+'_sha256'],'Approved artifact changed.')
    historical=Path('runs/conservative500-reviewed21')
    folders={'A0':(Path('runs/exploratory500-A0-reference'),historical/'A0-eval'),
        **{n+'_legacy':(Path('runs/conservative500-seed17')/(n+'-validation'),historical/(n+'-eval')) for n in ('A1','A2','A3')},
        'A2_fresh':(root/'A2-validation',root/'A2-validation-eval'),
        'A3_corrected':(root/'A3-validation',root/'A3-validation-eval')}
    models,raw,summaries,hashes={},{},{},{}
    reference=None;configurations={}
    for name,(images,folder) in folders.items():
        s=load_json(folder/'summary.json');g=load_json(images/'generation.json')
        rows=read_jsonl(folder/'metrics.jsonl'); ids=[r['sample_id'] for r in rows]
        require(len(rows)==len(set(ids))==24 and all(r['split']=='validation' for r in rows),'Wrong evaluation cohort.')
        require(g==s['generation'] and sha256(images/'generation.json')==s['generation_sha256'],'Generation metadata changed.')
        require(g['target_images_used_for_conditioning'] is False and g['stress']=='clean','Invalid prediction conditioning.')
        require(s['manual_index_sha256']==sha256(MANUAL) and s['manually_reviewed_count']==21,'Different scoring masks.')
        require(s['segmenter_revision']=='ec86afeba68e656629ccf47e0c8d2902f964917b','Evaluation teacher changed.')
        hashes[name]=verified_predictions(images);require(set(hashes[name])==set(ids),'Prediction cohort differs.')
        labels={r['sample_id']:tuple(r[k] for k in ('geo_group','target_building_pixels','valid_pixels','label_provenance')) for r in rows}
        if reference is None:reference=labels
        require(labels==reference,'Scoring targets differ across methods.')
        require(all(math.isfinite(r[k]) for r in rows for k in METRICS),'Non-finite evaluation.')
        means={k:grouped_mean(rows,k) for k in METRICS}
        require(all(math.isclose(means[k],s['group_weighted'][k],abs_tol=1e-12) for k in METRICS),'Aggregate differs from records.')
        if summaries:
            ref=next(iter(summaries.values()))
            for k in ('manifest_sha256','segmenter','segmenter_revision','empty_mask_convention'):
                require(s[k]==ref[k],'Evaluation setting changed: '+k)
            for k in ('style_sha256','manifest_sha256','train_manifest_sha256'):
                require(g[k]==ref['generation'][k],'Generation input changed: '+k)
            for k in ('size','render_seed','chunk_rows','checkpoint_rays','amp'):
                require(g['config'][k]==ref['generation']['config'][k],'Renderer changed: '+k)
            for sid in ids:
                require(sha256(folder/'targets'/(sid+'.png'))==sha256(historical/'A0-eval/targets'/(sid+'.png')),
                        'Real target pixels changed.')
        model=dict(metrics=means,strata=strata(rows),evaluation_sha256=sha256(folder/'summary.json'),
            metrics_sha256=sha256(folder/'metrics.jsonl'),generation_sha256=sha256(images/'generation.json'))
        if name in ('A2_fresh','A3_corrected'):
            n='A2' if name=='A2_fresh' else 'A3';checkpoint=root/(n+'-train/last.pt')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            train=load_json(root/(n+'-train/summary.json'));log=read_jsonl(root/(n+'-train/training.jsonl'))
            cfg=saved['config'];configurations[n]=cfg
            require(saved['step']==500 and cfg==g['config'] and g['trained_steps']==500,'Wrong checkpoint/step.')
            require(train['status']=='budget_complete' and train['step']==500 and
                    train['checkpoint_sha256']==g['checkpoint_sha256']==sha256(checkpoint),'Checkpoint/training incomplete.')
            require([r['step'] for r in log]==list(range(1,501)) and all(math.isfinite(v) for r in log for v in r.values()
                    if isinstance(v,(float,int))),'Incomplete/non-finite training log.')
            require(g['training_provenance']['pipeline_source_hash']==state['identity']['pipeline_source_hash'],
                    'Different training source.')
            model.update(checkpoint_sha256=sha256(checkpoint),finite_checkpoint_tensors={k:finite_values(saved[k]) for k in ('adapter','optimizer')},
                optimizer_steps=500,elapsed_seconds=train['elapsed_seconds'],peak_vram_mib=train['peak_vram_mib'])
        models[name]=model;raw[name]=rows;summaries[name]=s
    check_matched_configs(configurations['A2'],configurations['A3'])
    for key in ('train_data_digest','training_readiness'):
        require(summaries['A2_fresh']['generation'][key]==summaries['A3_corrected']['generation'][key],'Training inputs differ.')
    unchanged=sum(hashes['A2_fresh'][sid]==hashes['A2_legacy'][sid] for sid in hashes['A2_fresh'])
    control_equal=unchanged==24
    comparisons={}
    for control in ('A2_fresh','A3_legacy','A1_legacy','A0'):
        a,b=raw[control],raw['A3_corrected']
        comparisons['A3_corrected_vs_'+control]=dict(all24=delta(a,b),reviewed21=delta(
            [r for r in a if r['sample_id'] in index],[r for r in b if r['sample_id'] in index]),
            target_building=delta([r for r in a if r['target_building_pixels']>0],[r for r in b if r['target_building_pixels']>0]))
    result=dict(created_utc=utc_now(),study='exploratory_anchored_contour_correction',server_gate='not_eligible',
        fixed_final_step=500,seed=17,validation_views=24,reviewed_views=21,pseudo_label_views=3,
        primary='A3_corrected_vs_A2_fresh',models=models,comparisons=comparisons,
        fresh_A2_exact_historical_prediction_matches=unchanged,control_reproduced=control_equal,
        causal_attribution_requires_control_review=not control_equal,
        manual_index_sha256=sha256(MANUAL),runner_identity=state['identity']['pipeline_source_hash'],
        interpretation_pending=True,limitations=['One seed, 100 training views, already inspected development set.',
        'Training contours remain pseudo-labels; independent data QA incomplete.',
        'No novelty, audited generalization, or server-gate claim.'])
    output.mkdir(parents=True);local.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    lines=['# Fixed contour-correction comparison','','500 fresh steps per model; seed 17. All 24 development views retained.','',
        '| Model | Boundary | Reviewed boundary | IoU | LPIPS |','|---|---:|---:|---:|---:|']
    for n,m in models.items():
        a=m['metrics'];r=m['strata']['reviewed']['metrics']
        lines.append(f"| {n} | {a['boundary_error']:.6f} | {r['boundary_error']:.6f} | {a['building_iou']:.6f} | {a['lpips']:.6f} |")
    lines+=['',f'Fresh A2 exactly reproduces {unchanged}/24 historical prediction images.',
        'The primary contrast is corrected A3 versus fresh A2. See all geographic intervals and target-defined strata in [summary](summary.json).',
        'Interpretation and local visual inspection remain necessary; target coverage alone is not an improvement claim.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    order=sorted(index.items(),key=lambda item:item[1]['review_view_number'])
    for start in range(0,len(order),4):
        selected=order[start:start+4];canvas=Image.new('RGB',(5*192,len(selected)*220),'#17202a');draw=ImageDraw.Draw(canvas)
        for i,(sid,entry) in enumerate(selected):
            items=[('Target '+str(entry['review_view_number']),Path(entry['target_image']))]+[(n,folders[n][0]/(sid+'.png'))
                for n in ('A0','A2_fresh','A3_legacy','A3_corrected')]
            for col,(label,path) in enumerate(items):
                with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((192,192)),(col*192,i*220+23))
                draw.text((col*192+4,i*220+4),label,fill='white')
        canvas.save(local/f'page-{start//4+1}.png')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='runs/contour500-seed17')
    p.add_argument('--output',required=True);p.add_argument('--local-output',required=True)
    a=p.parse_args();report(a.root,a.output,a.local_output)
