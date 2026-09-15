"""Deterministic first-six field/geometry comparison for local scientific review."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from satground.common import ResearchError, read_jsonl, require_development, safe_data_path


def gallery(manifest, diagnostic, data_root, output, gd_predictions=None, gj_predictions=None):
    from satground.camera import crop_panorama
    rows = read_jsonl(manifest)
    require_development(rows, {'validation'})
    folders = [('A0', 'base'), ('Density only', 'density'), ('Appearance only', 'appearance'), ('Both', 'joint')]
    columns = ['Overhead', 'Real target'] + [n for n, _ in folders]
    if gd_predictions and gj_predictions:
        columns += ['GJ trained', 'GD trained']
    columns += ['Density radial change']
    tile, top, row_height = 180, 32, 204
    canvas = Image.new('RGB', (len(columns) * tile, top + 6 * row_height + 36), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(columns):
        draw.text((i * tile + 3, 8), label, fill='black')
    for i, row in enumerate(rows[:6]):
        sid = row['sample_id']
        paths = [Path(diagnostic) / folder / f'{sid}.png' for _, folder in folders]
        if gd_predictions and gj_predictions:
            paths += [Path(gj_predictions) / f'{sid}.png', Path(gd_predictions) / f'{sid}.png']
        with Image.open(safe_data_path(data_root, row['satellite'])) as image:
            images = [image.convert('RGB')]
        with Image.open(safe_data_path(data_root, row['panorama'])) as image:
            images.append(Image.fromarray(crop_panorama(image.convert('RGB'), row['yaw'], row['pitch'], row['fov'], row['size'])))
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        with np.load(Path(diagnostic) / 'geometry' / f'{sid}.npz') as data:
            change = data['density_radial'] - data['base_radial']
        # Fixed diverging range in MODEL coordinates for every view, never an
        # automatically normalized error map or a measured-depth comparison.
        scaled = np.clip(change / .04, -1, 1)
        rgb = np.ones((*scaled.shape, 3), dtype=np.float32)
        rgb[:, :, 0] -= np.maximum(-scaled, 0)
        rgb[:, :, 1] -= np.abs(scaled)
        rgb[:, :, 2] -= np.maximum(scaled, 0)
        images.append(Image.fromarray((rgb * 255).round().astype(np.uint8)))
        for j, image in enumerate(images):
            canvas.paste(image.resize((tile, tile)), (j * tile, top + i * row_height))
        draw.text((3, top + i * row_height + tile + 3), f'Fixed validation order: view {i + 1}', fill='black')
    draw.text((3, canvas.height - 24), 'Radial change: blue -0.04 / white 0 / red +0.04 model units. This is a change map, not a geometry error map.', fill='black')
    output = Path(output)
    if output.exists():
        raise ResearchError('Choose a new gallery filename to retain the prior review artifact.')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/learning100/validation.jsonl')
    parser.add_argument('--diagnostic', default='runs/fields-A3-seed17')
    parser.add_argument('--data-root', default='data/vigor')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gd-predictions')
    parser.add_argument('--gj-predictions')
    args = parser.parse_args()
    print(gallery(args.manifest, args.diagnostic, args.data_root, args.output, args.gd_predictions, args.gj_predictions))
