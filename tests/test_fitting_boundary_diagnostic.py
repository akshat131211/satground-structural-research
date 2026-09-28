"""Check artificial perimeter edges and CPU diagnostic provenance on synthetic masks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from diagnose_fitting_boundaries import boundary_counts, diagnose
from satground.common import ResearchError, load_json
from satground.metrics import building_metrics
from test_reviewed_fitting import completed_run_fixture


def test_full_building_has_only_primary_crop_edges():
    target = np.ones((16, 16), bool)
    valid = np.ones_like(target)
    result = boundary_counts(target, target, valid)
    assert result['primary_target_edge_pixels'] == result['target_perimeter_edge_pixels'] == 60
    assert result['target_nonperimeter_edge_pixels'] == 0
    assert result['nonperimeter_empty_case'] == 'both_empty'
    assert result['nonperimeter_boundary_error'] == 0


def test_interior_edges_match_primary_distance_when_no_mask_touches_border():
    target = np.zeros((20, 20), bool)
    target[4:14, 4:14] = True
    predicted = np.roll(target, 1, axis=1)
    valid = np.ones_like(target)
    result = boundary_counts(predicted, target, valid)
    assert result['target_perimeter_edge_pixels'] == 0
    assert result['nonperimeter_boundary_error'] == pytest.approx(building_metrics(predicted, target, valid)['boundary_error'])


def test_missing_nonperimeter_boundary_remains_penalized():
    target = np.zeros((16, 16), bool)
    target[3:12, 3:12] = True
    predicted = np.ones_like(target)
    result = boundary_counts(predicted, target, np.ones_like(target))
    assert result['nonperimeter_empty_case'] == 'one_empty'
    assert result['nonperimeter_boundary_error'] == 1


def test_ignored_boundary_does_not_reenter_diagnostic():
    target = np.zeros((16, 16), bool)
    target[:8] = True
    valid = np.ones_like(target)
    valid[6:10] = False
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    result = boundary_counts(target, target, valid)
    assert result['target_nonperimeter_edge_pixels'] == 0
    assert result['primary_target_edge_pixels'] == 0


def test_fresh_diagnostic_is_aggregate_only_and_does_not_change_primary_evidence(tmp_path):
    root = completed_run_fixture(tmp_path)
    bundle = tmp_path / 'review/bundle'
    result = diagnose(bundle, root, root / 'diagnostic.json', ['A0', 'A2', 'A3'])
    assert result['primary_metric_and_criterion_unchanged']
    assert not result['held_out_generalization_evidence'] and not result['server_gate_eligible']
    assert result['models']['A0']['primary_boundary_error'] == 0
    assert result == load_json(root / 'diagnostic.json')
    assert 'SYNTHETIC-3' not in (root / 'diagnostic.json').read_text()
    with pytest.raises(ResearchError, match='fresh'):
        diagnose(bundle, root, root / 'diagnostic.json', ['A0'])
