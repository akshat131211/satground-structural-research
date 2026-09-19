"""Keep imported experiments traceable and prevent a small noisy gain becoming a claim."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import autoresearch
from satground.common import ROOT, ResearchError, load_json, save_json, sha256


@pytest.fixture
def evidence():
    return load_json(ROOT / autoresearch.SUMMARY), load_json(ROOT / autoresearch.AUDIT)


def test_historical_small_improvement_stays_inconclusive(evidence):
    summary, audit = evidence
    result = autoresearch.assess_contour(summary, audit)
    assert result['measured_contrasts']['reviewed21']['relative_boundary_reduction'] == pytest.approx(0.016875345276960073)
    assert result['decision'] == 'inconclusive'
    assert result['quality_constraints_passed'] is True
    assert result['candidate_promoted'] is False
    assert result['server_gate_eligible'] is False
    assert result['kind'] == 'historical_import_no_new_training'
    assert result['empty_false_positive_fraction_change'] < 0
    assert 'inconclusive' in autoresearch.ledger_tsv(result)


@pytest.mark.parametrize('corruption', ['nan', 'boolean', 'cohort', 'labels', 'steps', 'scalar', 'interval', 'weighting'])
def test_bad_evidence_fails_closed(evidence, corruption):
    summary, audit = copy.deepcopy(evidence)
    contrast = summary['comparisons'][autoresearch.PRIMARY]['reviewed21']
    if corruption == 'nan':
        contrast['relative_boundary_reduction'] = float('nan')
    elif corruption == 'boolean':
        contrast['relative_boundary_reduction'] = True
    elif corruption == 'cohort':
        summary['reviewed_views'] = 20
    elif corruption == 'labels':
        audit['manual_index_sha256'] = 'different'
    elif corruption == 'steps':
        audit['training']['A3']['steps'] = 501
    elif corruption == 'scalar':
        contrast['relative_boundary_reduction'] = 0.9
    elif corruption == 'interval':
        contrast['boundary_bootstrap']['ci95'] = [0.1, -0.1]
    elif corruption == 'weighting':
        contrast['boundary_bootstrap']['weighting'] = 'independent_images'
    with pytest.raises(ResearchError):
        autoresearch.assess_contour(summary, audit)


def test_even_positive_intervals_do_not_promote_or_pass_server_gate(evidence):
    summary, audit = copy.deepcopy(evidence)
    for contrast in summary['comparisons'][autoresearch.PRIMARY].values():
        contrast['boundary_bootstrap']['ci95'] = [0.00001, 0.01]
    result = autoresearch.assess_contour(summary, audit)
    assert result['decision'] == 'exploratory_signal_requires_independent_confirmation'
    assert not result['candidate_promoted'] and not result['server_gate_eligible']


def test_empty_target_regression_is_not_hidden_by_boundary_gain(evidence):
    summary, audit = copy.deepcopy(evidence)
    summary['models']['A3_corrected']['strata']['target_empty']['group_weighted_false_positive_fraction'] = 0.1
    assert autoresearch.assess_contour(summary, audit)['decision'] == 'quality_regression'


@pytest.mark.parametrize('destination', ['../outside', 'reports/publish', 'runs/autoresearch', 'runs/autoresearch/../escape'])
def test_sessions_stay_in_private_run_directory(tmp_path, destination):
    with pytest.raises(ResearchError, match='beneath'):
        autoresearch.local_session(destination, tmp_path)


def test_existing_session_is_never_overwritten(tmp_path, monkeypatch):
    folder = tmp_path / 'runs/autoresearch/existing'
    folder.mkdir(parents=True)
    monkeypatch.setattr(autoresearch, 'local_session', lambda output: folder)
    with pytest.raises(ResearchError, match='immutable'):
        autoresearch.import_contour(folder)


def test_complete_form_does_not_bypass_review_verification(monkeypatch):
    monkeypatch.setattr(autoresearch, 'inspect_upstream', lambda root: {'revision': 'fixture'})
    monkeypatch.setattr(autoresearch, 'review_progress', lambda root: {'remaining': 0})
    result = autoresearch.status()
    assert result['state'] == 'requires_review_verification'
    assert not result['new_training_started'] and not result['training_launcher_implemented']


def test_ledger_reproduction_and_tamper_detection(tmp_path, evidence):
    summary, audit = evidence
    save_json(tmp_path / autoresearch.SUMMARY, summary)
    save_json(tmp_path / autoresearch.AUDIT, audit)
    output = tmp_path / 'runs/autoresearch/fixture'
    output.mkdir(parents=True)
    record = autoresearch.assess_contour(summary, audit)
    record['source_sha256'] = {name: sha256(tmp_path / name) for name in (autoresearch.SUMMARY, autoresearch.AUDIT)}
    save_json(output / 'record.json', record)
    save_json(output / 'completion-audit.json', audit)
    (output / 'results.tsv').write_text(autoresearch.ledger_tsv(record), encoding='utf-8')
    session = {'new_training_updates': 0,
               'files': {name: sha256(output / name) for name in ('record.json', 'results.tsv', 'completion-audit.json')}}
    save_json(output / 'session.json', session)
    assert autoresearch.verify_ledger(output, tmp_path)
    (output / 'results.tsv').write_text('fabricated improvement', encoding='utf-8')
    with pytest.raises(ResearchError, match='artifact changed'):
        autoresearch.verify_ledger(output, tmp_path)
    (output / 'results.tsv').write_text(autoresearch.ledger_tsv(record), encoding='utf-8')
    audit['reviewed_masks'] = 999
    save_json(tmp_path / autoresearch.AUDIT, audit)
    with pytest.raises(ResearchError, match='Historical evidence changed'):
        autoresearch.verify_ledger(output, tmp_path)


def test_upstream_pin_rejects_a_modified_checkout(tmp_path, monkeypatch):
    save_json(tmp_path / 'configs/autoresearch/upstream.json',
              {'local_path': 'external/reference', 'revision': 'pinned', 'git_blob_sha256': {}})
    replies = iter(['pinned\n', ' M train.py\n'])
    monkeypatch.setattr(autoresearch.subprocess, 'check_output', lambda *args, **kwargs: next(replies))
    with pytest.raises(ResearchError, match='checkout changed'):
        autoresearch.inspect_upstream(tmp_path)
