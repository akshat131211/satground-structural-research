"""Keep the lock reproducible without embedding a user's absolute checkout path."""
import subprocess
import sys
from pathlib import Path

lines = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
lines = ['-e .' if line.startswith('-e ') else line for line in lines]
Path('requirements-resolved.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
