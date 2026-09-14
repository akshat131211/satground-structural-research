"""Reject accidental credential, environment, dataset, or model-weight commits."""
import re
import subprocess
from pathlib import PurePosixPath

paths = subprocess.check_output(['git', 'ls-files', '-z'], text=True).split('\0')
forbidden = {'.venv', 'external', 'artifacts', 'data', 'dataset', 'runs', '.cache'}
patterns = [re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'), re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
            re.compile(rb'hf_[A-Za-z0-9]{25,}'), re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----')]
errors = []
for path in filter(None, paths):
    parts = PurePosixPath(path)
    if parts.parts[0] in forbidden or parts.name.startswith('.env') or parts.suffix in {'.pt', '.pth', '.safetensors', '.npz', '.pem', '.key'}:
        errors.append((path, 'local-only artifact'))
        continue
    content = subprocess.check_output(['git', 'show', ':' + path])
    if len(content) > 5 * 1024 * 1024:
        errors.append((path, 'unexpected large artifact'))
    if any(pattern.search(content) for pattern in patterns):
        errors.append((path, 'possible credential'))
for path, reason in errors:
    print(f'BLOCKED: {path}: {reason}')
if not errors:
    print(f'Publish check passed for {len(list(filter(None, paths)))} tracked files.')
raise SystemExit(1 if errors else 0)
