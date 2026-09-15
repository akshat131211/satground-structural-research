"""Do not confuse field interventions or unmatched image targets in reports."""
import copy
import importlib.util
from pathlib import Path

import pytest

from satground.common import ResearchError, save_json, write_jsonl

spec = importlib.util.spec_from_file_location('geometry_report', Path(__file__).parents[1] / 'scripts/report_geometry_pilot.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def summary():
    return dict(count=2, manifest_sha256='validation', segmenter='b1', segmenter_revision='pinned',
        group_weighted=dict(boundary_error=.2, building_iou=.5, lpips=.6, ssim=.2),
        generation=dict(scientific=True, target_images_used_for_conditioning=False,
            manifest_sha256='validation', style_sha256='train_style', train_manifest_sha256='train',
            train_data_digest='images', trained_steps=100, sampling='frozen_A0_coarse_plus_importance',
            provenance=dict(pipeline_source_hash='pipeline', model_revision='weights', code_revision='vendor', packages={})))


def test_rejects_mismatched_sampling_and_budget():
    a, b = summary(), summary()
    report.require_matched(a, b)
    b['generation']['sampling'] = 'adapted_proposal'
    with pytest.raises(ResearchError, match='sampling'):
        report.require_matched(a, b)
    b = summary()
    b['generation']['trained_steps'] = 10
    with pytest.raises(ResearchError, match='trained_steps'):
        report.require_matched(a, b)


def test_rejects_changed_metrics_or_target_conditioning(tmp_path):
    data = summary()
    rows = [dict(sample_id=str(i), geo_group=str(i), split='validation', **data['group_weighted']) for i in range(2)]
    save_json(tmp_path / 'summary.json', data)
    write_jsonl(tmp_path / 'metrics.jsonl', rows)
    report.read_evaluation(tmp_path)
    changed = copy.deepcopy(data)
    changed['group_weighted']['building_iou'] = .8
    save_json(tmp_path / 'summary.json', changed)
    with pytest.raises(ResearchError, match='aggregate'):
        report.read_evaluation(tmp_path)
    data['generation']['target_images_used_for_conditioning'] = True
    save_json(tmp_path / 'summary.json', data)
    with pytest.raises(ResearchError, match='target-free'):
        report.read_evaluation(tmp_path)
