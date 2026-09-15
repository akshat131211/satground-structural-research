"""Prevent unsupported comparisons when publishing preliminary measurements."""
import copy
import importlib.util
from pathlib import Path

import pytest

from satground.common import ResearchError

spec = importlib.util.spec_from_file_location('learning_report', Path(__file__).parents[1] / 'scripts/report_learning_check.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def evidence():
    rows = [dict(sample_id=str(i), geo_group=str(i), split='validation',
                 building_iou=.5, boundary_error=.2, lpips=.6, ssim=.2) for i in range(3)]
    summaries, records = {}, {}
    for name in ('A0', 'A1', 'A3'):
        cfg = dict.fromkeys(report.SHARED_CONFIG, 1)
        cfg.update(experiment=name, steps=100)
        gen = dict(config=cfg, scientific=True, target_images_used_for_conditioning=False,
                   manifest_sha256='validation', style_sha256='training_style', stress='clean',
                   train_manifest_sha256='train', trained_steps=0 if name == 'A0' else 100,
                   train_data_digest='rgb_hashes', provenance=dict(pipeline_source_hash='source',
                   code_revision='code', model_revision='weights', packages={'torch': 'pinned'}))
        summaries[name] = dict(generation=gen, count=3, manifest_sha256='validation',
            segmenter_revision='b1', segmenter='independent_b1', training_segmenter_different=True,
            group_weighted={k: rows[0][k] for k in report.METRICS})
        records[name] = copy.deepcopy(rows)
    return summaries, records


def test_rejects_mismatched_training_budget_and_target_conditioning():
    summaries, records = evidence()
    report.matched_evidence(summaries, records)
    summaries['A3']['generation']['trained_steps'] = 50
    with pytest.raises(ResearchError, match='budget'):
        report.matched_evidence(summaries, records)
    summaries, records = evidence()
    summaries['A3']['generation']['target_images_used_for_conditioning'] = True
    with pytest.raises(ResearchError, match='target-free'):
        report.matched_evidence(summaries, records)


def test_rejects_audit_rows_and_changed_sample_pairing():
    summaries, records = evidence()
    records['A3'][0]['split'] = 'audit'
    with pytest.raises(ResearchError, match='validation data only'):
        report.matched_evidence(summaries, records)
    records['A3'][0]['split'] = 'validation'
    records['A3'][0]['sample_id'] = 'different_location'
    with pytest.raises(ResearchError, match='sample IDs'):
        report.matched_evidence(summaries, records)


def test_rejects_changed_summary_or_model_source():
    summaries, records = evidence()
    summaries['A3']['group_weighted']['boundary_error'] = .1
    with pytest.raises(ResearchError, match='aggregate'):
        report.matched_evidence(summaries, records)
    summaries, records = evidence()
    summaries['A3']['generation']['provenance']['model_revision'] = 'other_weights'
    with pytest.raises(ResearchError, match='source/model'):
        report.matched_evidence(summaries, records)
