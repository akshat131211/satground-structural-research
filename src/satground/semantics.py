"""Frozen Cityscapes semantic models; pseudo-label provenance stays explicit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera import crop_panorama
from .common import (ResearchError, read_jsonl, require_development, safe_data_path,
                     save_json, sha256)
from .manifests import require_files

TRAIN_SEGMENTER = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
EVAL_SEGMENTER = 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024'
SEGMENTER_REVISIONS = {
    TRAIN_SEGMENTER: '21b3847fae21ddee674abd31129307b6a1235bd9',
    EVAL_SEGMENTER: 'ec86afeba68e656629ccf47e0c8d2902f964917b',
}


class Segmenter:
    def __init__(self, model_id=TRAIN_SEGMENTER, revision=None, device='cuda', input_size=256):
        from huggingface_hub import HfApi
        from transformers import SegformerForSemanticSegmentation
        self.model_id = model_id
        self.revision = revision or SEGMENTER_REVISIONS.get(model_id) or HfApi().model_info(model_id).sha
        self.device, self.input_size = device, input_size
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id, revision=self.revision, weights_only=True,
            use_safetensors=False if model_id in SEGMENTER_REVISIONS else None).to(device).eval().requires_grad_(False)
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if labels.get(2) != 'building' or labels.get(10) != 'sky':
            raise ResearchError('Expected Cityscapes class IDs, received a different semantic taxonomy.')

    def probabilities(self, rgb):
        # Keep gradients with respect to the generated RGB, but never update semantic weights.
        size = rgb.shape[-2:]
        x = F.interpolate(rgb, (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        logits = self.model(pixel_values=(x - mean) / std).logits
        return F.interpolate(logits, size, mode='bilinear', align_corners=False).softmax(dim=1)


def rgb_tensor(image, device='cpu'):
    array = np.asarray(image, dtype=np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def make_labels(manifest, data_root, output, model_id=TRAIN_SEGMENTER, revision=None, confidence=.7):
    rows = read_jsonl(manifest)
    require_development(rows, {'train', 'validation'})
    require_files(rows, data_root)
    segmenter = Segmenter(model_id, revision)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        panorama = Image.open(safe_data_path(data_root, row['panorama'])).convert('RGB')
        target = crop_panorama(panorama, row['yaw'], row['pitch'], row['fov'], row['size'])
        with torch.no_grad():
            probs = segmenter.probabilities(rgb_tensor(target, 'cuda'))
            certainty, labels = probs.max(dim=1)
            valid = (certainty >= confidence) & (labels < 11)
        np.savez_compressed(out / f"{row['sample_id']}.npz", building=(labels[0] == 2).cpu().numpy(),
                            valid=valid[0].cpu().numpy(), classes=labels[0].cpu().numpy().astype(np.uint8),
                            confidence=certainty[0].cpu().numpy().astype(np.float32))
    source_paths = sorted({r['panorama'] for r in rows})
    record = dict(kind='pseudo_labels', human_verified=False, model_id=model_id,
                  revision=segmenter.revision, confidence_threshold=confidence,
                  manifest_sha256=sha256(manifest), count=len(rows), input_size=segmenter.input_size,
                  ignored_classes=list(range(11, 19)), output_size=rows[0]['size'],
                  target_sha256={p: sha256(safe_data_path(data_root, p)) for p in source_paths},
                  label_sha256={r['sample_id']: sha256(out / f"{r['sample_id']}.npz") for r in rows})
    save_json(out / 'provenance.json', record)
    return record


def make_style(manifest, data_root, output):
    """Mean of per-panorama histograms using supplied TRAINING sky masks only."""
    rows = read_jsonl(manifest)
    require_development(rows, {'train'})
    require_files(rows, data_root)
    histograms, sources = [], []
    for row in {r['panorama']: r for r in rows}.values():
        path = safe_data_path(data_root, row['sky_mask'])
        if not path.is_file():
            raise ResearchError(f'Training sky mask missing: {path}. Obtain the public city sky-mask supplement.')
        with Image.open(safe_data_path(data_root, row['panorama'])) as panorama, Image.open(path) as mask:
            values = training_sky_histogram(panorama, mask)
        if values is not None:
            histograms.append(values)
            sources.append(dict(panorama=row['panorama'], image_sha256=sha256(safe_data_path(data_root, row['panorama'])),
                                mask_sha256=sha256(path)))
    if not histograms:
        raise ResearchError('No usable training sky pixels. Cannot create a scientific illumination code.')
    record = dict(source_split='train', synthetic=False, manifest_sha256=sha256(manifest),
                  histogram=np.mean(histograms, axis=0).tolist(), panorama_count=len(histograms), sources=sources,
                  method='mean_per_panorama_RGB_100bins_drop_first10_pinned_Sat3DGen_resize512x128',
                  preprocessing={'size': [512, 128], 'rgb': 'PIL_bicubic', 'mask': 'nearest'})
    save_json(output, record)
    return {k: v for k, v in record.items() if k not in ('sources', 'histogram')}


def training_sky_histogram(panorama, mask):
    """Match the pinned release's illumination preprocessing before averaging."""
    if panorama.size != mask.size:
        raise ResearchError('Sky mask is not aligned with its panorama.')
    rgb = np.asarray(panorama.convert('RGB').resize((512, 128), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    sky = np.asarray(mask.convert('L').resize((512, 128), Image.Resampling.NEAREST), dtype=np.float32) / 255
    image = (rgb * sky[..., None]) * 2 - 1
    values = []
    for channel in image.transpose(2, 0, 1):
        histogram = np.histogram(channel, bins=100, range=(-1, 1))[0][10:].astype(np.float64)
        if histogram.sum() <= 0:
            return None
        values.extend(histogram / histogram.sum())
    return values
