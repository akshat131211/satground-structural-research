"""Review selection must preserve split boundaries and avoid repeated scenes."""
import importlib.util
from pathlib import Path

import pytest

from satground.common import ResearchError

spec = importlib.util.spec_from_file_location('review_packet', Path(__file__).parents[1] / 'scripts/prepare_review_packet.py')
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_selection_is_deterministic_unique_and_development_only():
    rows = [dict(sample_id=f'{split}{i}{yaw}', city='Chicago', split=split, geo_group=f'{split}{i}',
                 panorama=f'{split}/pano{i}', satellite=f'{split}/sat{i}', yaw=yaw)
            for split in ('train', 'validation') for i in range(5) for yaw in range(4)]
    result = packet.select_views(rows, per_split=3)
    assert result == packet.select_views(list(reversed(rows)), per_split=3)
    assert len(result) == len({r['panorama'] for r in result}) == len({r['satellite'] for r in result}) == 6
    assert sum(r['split'] == 'train' for r in result) == 3
    with pytest.raises(ResearchError, match='Insufficient unique'):
        packet.select_views(rows, per_split=6)
    rows[0]['split'] = 'audit'
    with pytest.raises(ResearchError, match='not allowed'):
        packet.select_views(rows, per_split=3)
