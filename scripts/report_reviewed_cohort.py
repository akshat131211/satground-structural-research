"""Report a versioned reviewed subset while preserving the original model scores."""
import argparse

from satground.review_reporting import report_reviewed_cohort


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--evaluation-tag', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runs-root', default='runs')
    args = parser.parse_args()
    result = report_reviewed_cohort(args.index, args.evaluation_tag, args.output, args.runs_root)
    print('Verified fixed-prediction diagnostic:', result['reviewed_views'], 'reviewed views')
