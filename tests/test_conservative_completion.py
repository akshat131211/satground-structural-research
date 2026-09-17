import importlib.util
from pathlib import Path

import pytest

from satground.common import ResearchError, save_json, sha256


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('conservative_completion', scripts / 'audit_conservative500.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_audit_preserves_existing_output(module, tmp_path):
    output = tmp_path / 'audit.json'
    output.write_text('preserved')
    with pytest.raises(ResearchError, match='fresh audit paths'):
        module.audit(output, tmp_path / 'work')
    assert output.read_text() == 'preserved'


def test_forged_completion_without_stages_is_rejected(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_json('runs/conservative500-seed17/status.json', dict(status='complete', completed=[]))
    with pytest.raises(ResearchError, match='All nine stages'):
        module.audit('audit.json', 'work')
    assert not Path('audit.json').exists()


def test_changed_pinned_input_blocks_audit(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pin = Path('config.yaml')
    pin.write_text('original')
    state = dict(status='complete', completed=[n + s for n in ('A1', 'A2', 'A3')
                 for s in ('-train', '-validation', '-validation-eval')],
                 identity=dict(files={str(pin): sha256(pin)}))
    save_json('runs/conservative500-seed17/status.json', state)
    pin.write_text('modified')
    with pytest.raises(ResearchError, match='Pinned inputs changed'):
        module.audit('audit.json', 'work')
    assert not Path('work').exists()
