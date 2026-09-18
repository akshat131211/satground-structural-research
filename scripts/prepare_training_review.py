"""Prepare a fixed, training-only context and mask review; no model inference."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_pairing_context import geometry
from satground.camera import crop_panorama
from satground.common import (ResearchError, TRAIN_CITIES, load_json, read_jsonl,
    require_development, safe_data_path, save_json, sha256, utc_now, write_jsonl)
from satground.editor import mask_card
from satground.manifests import assert_disjoint

SELECTION = ('Four distinct geographic groups per training city, ordered by SHA256 of '
    'training-review12-v1:group; one sample per group ordered by SHA256 of '
    'training-review12-v1:sample. Distinct tiles and panoramas; no image, label or score selection.')


def rank(value):
    return hashlib.sha256(('training-review12-v1:'+value).encode()).hexdigest()


def select_training(rows):
    require_development(rows, {'train'})
    chosen, tiles, panoramas, groups = [], set(), set(), set()
    for city in sorted(TRAIN_CITIES):
        selected = 0
        candidates = [r for r in rows if r['city']==city]
        for group in sorted({r['geo_group'] for r in candidates}, key=rank):
            if group in groups:
                continue
            eligible = [r for r in candidates if r['geo_group']==group and
                        r['satellite'] not in tiles and r['panorama'] not in panoramas]
            if not eligible:
                continue
            row = min(eligible,key=lambda r:rank(r['sample_id']))
            chosen.append(row); tiles.add(row['satellite']); panoramas.add(row['panorama']); groups.add(group)
            selected += 1
            if selected==4:
                break
        if selected!=4:
            raise ResearchError('Need four distinct eligible training groups per city.')
    return chosen


def context_card(row, satellite, panorama, number):
    if satellite.size!=(640,640) or (row['pitch'],row['fov'],row['size'])!=(0,90,256):
        raise ResearchError('Context drawing requires the released 640-pixel tile and 256-pixel 90-degree view.')
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    page=Image.new('RGB',(1280,770),'#121922');draw=ImageDraw.Draw(page)
    draw.text((14,10),f'Training view {number}/12 | Satellite and real ground directions | No model predictions',fill='white',font=font)
    overlay=satellite.convert('RGBA');tint=Image.new('RGBA',satellite.size);mark=ImageDraw.Draw(tint)
    origin,directions,exits=geometry(row,row['yaw'])
    mark.polygon([tuple(origin),tuple(origin+directions[0]*2000),tuple(origin+directions[2]*2000)],fill=(255,216,89,40))
    for edge in (exits[0],exits[2]):mark.line([tuple(origin),tuple(edge)],fill=(255,216,89,255),width=3)
    overlay=Image.alpha_composite(overlay,tint).convert('RGB');mark=ImageDraw.Draw(overlay)
    yaws=(row['yaw'],*(((row['yaw']+x+180)%360)-180 for x in (90,180,-90)))
    crops=[]
    for i,(yaw,color) in enumerate(zip(yaws,('#ffd859','#ff7878','#74dfa9','#70b7ff'))):
        origin,directions,_=geometry(row,yaw);d=directions[1];end=origin+d*115;normal=np.array([-d[1],d[0]])
        mark.line([tuple(origin),tuple(end)],fill=color,width=4)
        mark.polygon([tuple(end),tuple(end-d*15+normal*6),tuple(end-d*15-normal*6)],fill=color)
        mark.text(tuple(end+normal*8),str(i+1),font=font,fill=color,stroke_width=1,stroke_fill='black')
        crop=crop_panorama(panorama,yaw,0,90,256);crops.append(crop)
        x,y=650+(i%2)*310,48+(i//2)*320
        draw.text((x,y),f'{i+1}: '+('MASK TARGET' if i==0 else 'Other direction'),fill=color,font=font)
        page.paste(Image.fromarray(crop).resize((292,292),Image.Resampling.NEAREST),(x,y+23))
    x,y=origin;mark.ellipse((x-7,y-7,x+7,y+7),fill='white',outline='black',width=2)
    page.paste(overlay,(0,48))
    draw.text((12,702),'White dot: released camera. Yellow wedge/arrow 1: mask target. Other arrows: context only.',fill='white',font=font)
    draw.text((12,727),'Heading, metric scale and building identity are unverified. Wedge is direction, not visibility.',fill='#bbc7d5',font=font)
    return page,crops


def prepare(output):
    out=Path(output)
    if out.exists():raise ResearchError('Preserve existing reviews; use a fresh output directory.')
    manifest=Path('data/manifests/learning100/train.jsonl');rows=read_jsonl(manifest)
    selected=select_training(rows)
    validation=read_jsonl('data/manifests/learning100/validation.jsonl')
    require_development(validation,{'validation'});assert_disjoint(rows+validation)
    labels=Path('data/labels/contour100-v1');provenance=load_json(labels/'provenance.json')
    if provenance['manifest_sha256']!=sha256(manifest):raise ResearchError('Training labels belong to another manifest.')
    frozen=load_json('runs/contour500-seed17/status.json')['identity']['files']
    if any(sha256(p)!=h for p,h in frozen.items()):raise ResearchError('Completed comparison inputs changed.')
    for name in ('targets','proposals','cards','context','decisions'):(out/name).mkdir(parents=True)
    samples=[]
    for number,row in enumerate(selected,1):
        sid=row['sample_id'];label_path=labels/(sid+'.npz')
        sat=safe_data_path('data/vigor',row['satellite']);pano=safe_data_path('data/vigor',row['panorama'])
        if sha256(label_path)!=provenance['label_sha256'][sid] or sha256(pano)!=provenance['target_sha256'][row['panorama']]:
            raise ResearchError('Cached label/source target identity changed.')
        with Image.open(sat) as image:satellite=image.convert('RGB')
        with Image.open(pano) as image:panorama=image.convert('RGB')
        card,crops=context_card(row,satellite,panorama,number);target=crops[0]
        paths=dict(target_image=f'targets/{sid}.png',context_image=f'context/view-{number:02d}.png',
                   satellite_image=f'context/view-{number:02d}-satellite.png')
        Image.fromarray(target).save(out/paths['target_image']);card.save(out/paths['context_image']);satellite.save(out/paths['satellite_image'])
        with np.load(label_path) as data:
            for component in ('building','valid'):
                mask=data[component].astype(bool)
                paths[component+'_mask']=f'proposals/{sid}-{component}.png'
                paths[component+'_card']=f'cards/view-{number:02d}-{component}.png'
                Image.fromarray(mask.astype(np.uint8)*255).save(out/paths[component+'_mask'])
                mask_card(target,mask,component).save(out/paths[component+'_card'])
        samples.append(dict(sample_id=sid,number=number,reviewed=False,**paths,
            **{k+'_sha256':sha256(out/p) for k,p in paths.items()},source_label_sha256=sha256(label_path),
            source_satellite_sha256=sha256(sat),source_panorama_sha256=sha256(pano)))
    write_jsonl(out/'manifest.jsonl',selected)
    record=dict(created_utc=utc_now(),samples=samples,purpose='training_learnability_review',selection=SELECTION,
        source_manifest_sha256=sha256(manifest),manifest_sha256=sha256(out/'manifest.jsonl'),
        source_labels_sha256=sha256(labels/'provenance.json'),proposal_model=provenance['model_id'],
        proposal_revision=provenance['revision'],proposal_input_size=provenance['input_size'],
        proposal_provenance='unchanged_cached_training_pseudo_labels',human_review_complete=False,reviewed_count=0,
        review_blinded_to_model_predictions=False,
        blinding_note='This surface excludes generated images. Prior assistant inspection of some training proposals is possible; no full blinding claim.',
        training_eligible=False,review_role='training_only',evaluation_masks_reused=False,
        context_note='Released camera directions are aids, not certified heading/scale/visibility or correspondence.',
        source_images_used_for_model_conditioning=False)
    save_json(out/'packet.json',record)
    public=dict(created_utc=utc_now(),status='awaiting_human_review',training_views=12,
        geographic_groups=len({r['geo_group'] for r in selected}),city_counts=dict(Counter(r['city'] for r in selected)),
        unique_tiles=len({r['satellite'] for r in selected}),unique_panoramas=len({r['panorama'] for r in selected}),
        selection=SELECTION,packet_sha256=sha256(out/'packet.json'),manifest_sha256=sha256(out/'manifest.jsonl'),
        reviewed_count=0,training_eligible=False,new_training_started=False,model_inference_run=False,
        masks_from_cached_training_labels=True,evaluation_masks_unchanged=True,sealed_images_opened=False)
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='data/review/training12-v1')
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    if Path(args.report).exists():raise ResearchError('Preserve the previous aggregate report; use a fresh report path.')
    result=prepare(args.output);save_json(args.report,result);print(result)
