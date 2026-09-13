"""Run verification and save compact, shareable evidence without imagery or paths."""
from pathlib import Path
import os
import subprocess
import sys

from satground.common import provenance, save_json

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, SATGROUND_GPU_TESTS='1')
commands = [
    [sys.executable, '-m', 'pytest', '-q'],
    [sys.executable, '-m', 'pip', 'check'],
]
checks = []
for command in commands:
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='')
    checks.append(dict(command=' '.join(['python', *command[1:]]), exit_code=result.returncode,
                       output=result.stdout.strip()))
result = subprocess.run([sys.executable, '-m', 'satground', 'train', '--config', 'configs/A3.yaml',
                         '--data-root', 'data/vigor', '--output', 'runs/real-data-must-block'],
                        cwd=root, text=True, capture_output=True)
blocked = result.returncode == 2 and 'Original VIGOR RGB images are required' in result.stderr
checks.append(dict(command='real-data training rejects absent RGB', passed=blocked,
                   exit_code=result.returncode, output='Missing original VIGOR images; no training run created.' if blocked else 'Unexpected outcome'))
record = dict(provenance=provenance(), checks=checks,
              status='passed' if all(c.get('passed', c['exit_code'] == 0) for c in checks) else 'failed')
save_json(root / 'reports' / 'verification.json', record)
raise SystemExit(0 if record['status'] == 'passed' else 1)
