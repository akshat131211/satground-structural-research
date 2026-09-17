"""Open one numbered target from the additional review batch in the existing editor."""
import argparse
from pathlib import Path
import subprocess
import sys

from satground.common import ROOT, ResearchError, load_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--packet', default='data/review/checkpoint-review18')
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    packet = Path(args.packet)
    candidates = [s for s in load_json(packet / 'packet.json')['samples'] if s['number'] == args.view]
    if len(candidates) != 1:
        raise ResearchError('Choose an available view number from this packet.')
    command = [sys.executable, '-u', str(ROOT / 'scripts/edit_masks.py'), '--packet', str(packet),
               '--sample-id', candidates[0]['sample_id']]
    if args.open_browser:
        command.append('--open-browser')
    subprocess.run(command, cwd=ROOT, check=True)
