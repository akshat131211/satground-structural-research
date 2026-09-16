import importlib.util
import json
from pathlib import Path

import pytest
import torch

from satground.common import ResearchError, sha256


@pytest.fixture
def audit_module(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('completion_audit', scripts / 'audit_exploratory500.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_tampering_is_rejected(audit_module, tmp_path):
    image = tmp_path / 'prediction.png'
    image.write_bytes(b'original saved image bytes')
    record = {'sample_id': 'test', 'prediction': image.name, 'sha256': sha256(image)}
    (tmp_path / 'predictions.jsonl').write_text(json.dumps(record) + '\n')
    assert audit_module.verified_predictions(tmp_path) == {'test': record['sha256']}
    image.write_bytes(b'changed after evaluation')
    with pytest.raises(ResearchError, match='Prediction file changed'):
        audit_module.verified_predictions(tmp_path)


def test_duplicate_prediction_is_rejected(audit_module, tmp_path):
    image = tmp_path / 'prediction.png'
    image.write_bytes(b'image bytes')
    record = {'sample_id': 'test', 'prediction': image.name, 'sha256': sha256(image)}
    (tmp_path / 'predictions.jsonl').write_text((json.dumps(record) + '\n') * 2)
    with pytest.raises(ResearchError, match='Duplicate prediction'):
        audit_module.verified_predictions(tmp_path)


def test_nested_nonfinite_optimizer_is_rejected(audit_module):
    with pytest.raises(ResearchError, match='Non-finite checkpoint tensor'):
        audit_module.finite_values({'state': {0: {'exp_avg': torch.tensor([0.0, float('nan')])}}})


def test_existing_report_is_preserved(audit_module, tmp_path):
    output = tmp_path / 'report.json'
    output.write_text('original report')
    with pytest.raises(ResearchError, match='fresh audit output'):
        audit_module.audit(output, tmp_path / 'work')
    assert output.read_text() == 'original report'
