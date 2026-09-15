"""Export one exact-image evaluation mask pair from explicit human decisions."""
import argparse

from satground.annotations import finalize_review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'sample-id', 'decision', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(finalize_review(args.packet, args.sample_id, args.decision, args.output))
