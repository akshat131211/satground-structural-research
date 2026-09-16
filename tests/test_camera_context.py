import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('camera_context', Path(__file__).parents[1] / 'scripts/prepare_camera_context.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('yaw,expected', [(0, [0,-1]), (90,[1,0]), (180,[0,1]), (-90,[-1,0])])
def test_cardinal_overlay_directions(yaw, expected):
    origin, directions, _ = module.geometry(dict(h_offset=-92,w_offset=19), yaw)
    np.testing.assert_array_equal(origin, [339,228])
    np.testing.assert_allclose(directions[1], expected, atol=1e-7)


def test_view_three_forward_coverage_is_not_metric_distance():
    origin, directions, exits = module.geometry(dict(h_offset=-92,w_offset=19), 0)
    np.testing.assert_allclose(exits[1], [339,0], atol=1e-7)
    assert np.linalg.norm(exits[1]-origin) == pytest.approx(228)
    np.testing.assert_allclose(module.ray_exit(origin,directions[1],(-80,720)),[339,-80],atol=1e-7)


def test_edge_origin_looking_inward_crosses_tile():
    np.testing.assert_array_equal(module.ray_exit([0,320],[1,0]),[640,320])
    np.testing.assert_array_equal(module.ray_exit([0,320],[-1,0]),[0,320])
