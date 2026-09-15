import importlib.util
from pathlib import Path

import pytest

from satground.common import ResearchError

spec = importlib.util.spec_from_file_location('multiview_review', Path(__file__).parents[1] / 'scripts/prepare_multiview_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def rows():
    return [dict(sample_id=f'{city}-{pano}-{yaw}', city=city, geo_group=f'{city}-one',
                 satellite=f'{city}/tile.png', panorama=f'{city}/pano{pano}.jpg', split='train',
                 yaw=yaw, h_offset=pano * 10, w_offset=0)
            for city in ('Chicago', 'NewYork', 'SanFrancisco')
            for pano in range(3) for yaw in (0, 90, 180, 270)]


def test_selects_distinct_positions_and_preserves_all_directions():
    chosen, inventory = review.select_tiles(rows(), per_city=1)
    reverse, _ = review.select_tiles(list(reversed(rows())), per_city=1)
    assert chosen == reverse
    assert inventory['eligible_multiview_tiles'] == 3
    equivalent = rows()
    for row in equivalent:
        if row['yaw'] == 270:
            row['yaw'] = -90
    wrapped, _ = review.select_tiles(equivalent, per_city=1)
    assert len(wrapped) == len(chosen)
    assert any(r['yaw'] == -90 for r in wrapped[0]['rows'])
    for tile in chosen:
        assert len(tile['rows']) == 8
        assert {r['h_offset'] for r in tile['rows']} == {0, 20}
        assert tile['separation_offset_pixels'] == 20


def test_rejects_nontraining_or_identical_camera_positions():
    examples = rows()
    examples[0]['split'] = 'validation'
    with pytest.raises(ResearchError, match='not allowed'):
        review.select_tiles(examples, per_city=1)
    examples = rows()
    for row in examples:
        row['h_offset'] = 0
    with pytest.raises(ResearchError, match='two camera positions'):
        review.select_tiles(examples, per_city=1)
