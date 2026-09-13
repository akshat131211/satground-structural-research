"""Preflight for real data; short learning checks do not certify the main study."""
from pathlib import Path

from .common import ResearchError, load_json, read_jsonl, sha256


def require_training_readiness(config, rows, last_step):
    validation_path = config.get('data_validation', 'reports/data-readiness.json')
    report = load_json(validation_path)
    if report['status'] != 'ready_for_review' or report['missing_count'] or report['invalid'] or report['cross_split_duplicate_hashes']:
        raise ResearchError('RGB verification is incomplete or has unresolved file/duplicate errors.')
    all_manifest = config.get('all_manifest', 'data/manifests/pilot/all.jsonl')
    if report['manifest_sha256'] != sha256(all_manifest):
        raise ResearchError('Data verification belongs to a different complete split manifest.')
    allowed = {r['sample_id']: r for r in read_jsonl(all_manifest)}
    if any(r['sample_id'] not in allowed or r != allowed[r['sample_id']] for r in rows):
        raise ResearchError('Training/validation samples differ from the verified split set.')
    if last_step <= 100:
        return {'stage': 'preliminary_learning_check', 'human_qa_may_be_pending': True}
    review = load_json(config.get('data_qa', 'data/qa/review.json'))
    required = ('footprint_bound_verified', 'pairing_review_complete', 'near_duplicate_review_complete')
    if any(review.get(key) is not True for key in required) or not review.get('reviewer') or not review.get('reviewed_utc'):
        raise ResearchError('Main training requires documented footprint, pairing, and duplicate review.')
    if review.get('data_validation_sha256') != sha256(validation_path):
        raise ResearchError('Data QA refers to an older validation report.')
    return {'stage': 'main_pilot', 'data_qa_sha256': sha256(config.get('data_qa', 'data/qa/review.json'))}
