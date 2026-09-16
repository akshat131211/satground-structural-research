import pytest

from satground.common import ResearchError, save_json, sha256, write_jsonl
from satground.readiness import require_training_readiness


def fixture_data(tmp_path):
    row = dict(sample_id='one', split='train')
    manifest, report = tmp_path / 'all.jsonl', tmp_path / 'data.json'
    write_jsonl(manifest, [row])
    record = dict(status='ready_for_review', missing_count=0, invalid=[],
                  cross_split_duplicate_hashes=[], manifest_sha256=sha256(manifest))
    save_json(report, record)
    config = dict(all_manifest=str(manifest), data_validation=str(report),
                  data_qa=str(tmp_path / 'missing-review.json'), data_review_mode='exploratory',
                  exploratory_reason='Test explicit exploratory configuration.')
    return config, row, record


def test_exploratory_allows_longer_run_without_fabricating_qa(tmp_path):
    config, row, _ = fixture_data(tmp_path)
    result = require_training_readiness(config, [row], 500)
    assert result['stage'] == 'exploratory_pending_data_review'
    assert result['human_qa_may_be_pending'] is True
    assert result['main_study_eligible'] is False
    assert 'pairing_review_complete' in result['pending_checks']
    assert result['data_qa_sha256'] is None
    assert not (tmp_path / 'missing-review.json').exists()


@pytest.mark.parametrize('change', ['missing', 'invalid', 'duplicate', 'manifest', 'row'])
def test_exploratory_keeps_data_integrity_checks(tmp_path, change):
    config, row, record = fixture_data(tmp_path)
    if change == 'missing':
        record['missing_count'] = 1
    elif change == 'invalid':
        record['invalid'] = ['bad-image']
    elif change == 'duplicate':
        record['cross_split_duplicate_hashes'] = ['same-image']
    elif change == 'manifest':
        record['manifest_sha256'] = 'stale'
    else:
        row['split'] = 'audit'
    save_json(config['data_validation'], record)
    with pytest.raises(ResearchError):
        require_training_readiness(config, [row], 500)


def test_exploratory_needs_explicit_mode_and_reason(tmp_path):
    config, row, _ = fixture_data(tmp_path)
    config['data_review_mode'] = 'typo'
    with pytest.raises(ResearchError, match='Unknown'):
        require_training_readiness(config, [row], 500)
    config['data_review_mode'] = 'exploratory'
    config['exploratory_reason'] = ''
    with pytest.raises(ResearchError, match='reason'):
        require_training_readiness(config, [row], 500)


def test_server_report_rejects_exploratory_before_claims(tmp_path):
    from satground.evaluation import make_report
    summary = dict(generation=dict(config=dict(data_review_mode='exploratory')))
    save_json(tmp_path / 'summary.json', summary)
    save_json(tmp_path / 'spec.json', dict(comparisons=[dict(
        control_metrics=str(tmp_path / 'metrics.jsonl'), proposed_metrics=str(tmp_path / 'metrics.jsonl'))]))
    with pytest.raises(ResearchError, match='Exploratory training'):
        make_report(tmp_path / 'spec.json', tmp_path / 'output')
