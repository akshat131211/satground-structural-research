"""Prepare a deterministic local training-only contour proposal inspection packet."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from satground.camera import crop_panorama
from satground.common import read_jsonl, require_development, safe_data_path, sha256
from satground.contours import contour_support


def prepare(output):
    output=Path(output)
    if output.exists():raise RuntimeError('Use a fresh inspection output.')
    rows=sorted(read_jsonl('data/manifests/learning100/train.jsonl'),key=lambda r:r['sample_id'])
    require_development(rows,{'train'})
    # Fixed ordering and even spacing; never select examples by validation performance.
    selected=[rows[i] for i in np.linspace(0,len(rows)-1,8,dtype=int)]
    output.mkdir(parents=True)
    records=[]
    for page in range(2):
        canvas=Image.new('RGB',(3*256,4*285),'#17202a');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(selected[page*4:(page+1)*4]):
            with Image.open(safe_data_path('data/vigor',row['panorama'])) as image:
                target=crop_panorama(image.convert('RGB'),row['yaw'],row['pitch'],row['fov'],row['size'])
            with np.load(Path('data/labels/contour100-v1')/(row['sample_id']+'.npz')) as label:
                s=contour_support(*(label[k] for k in ('building','valid','classes','confidence')))
                overlay=target.copy(); b=label['building'].astype(bool)
                overlay[b]=(overlay[b]*.6+np.array([255,60,70])*.4).astype(np.uint8)
                support=target.copy();support[s['added_positive']]=[0,255,255]
                support[s['target_boundary'] & s['legacy_valid']]=[255,230,0]
            number=page*4+j+1
            for col,(name,array) in enumerate([('Real training target',target),('Red: pseudo building',overlay),('Cyan: added contour',support)]):
                draw.text((col*256+4,j*285+5),f'{number}: {name}',fill='white')
                canvas.paste(Image.fromarray(array),(col*256,j*285+25))
            records.append(dict(inspection_number=number,sample_id=row['sample_id']))
        canvas.save(output/f'page-{page+1}.png')
    (output/'selection.json').write_text(json.dumps(dict(selection='8 evenly spaced sorted training IDs',
        training_manifest_sha256=sha256('data/manifests/learning100/train.jsonl'),views=records,
        human_verified=False),indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':prepare('data/review/contour-training-inspection-v1')
