from __future__ import annotations

import argparse
import json
import sys

from .common import ResearchError, load_json, read_jsonl, save_json


def parser():
    p = argparse.ArgumentParser(prog='satground', description='Single-overhead structural ground-view research pipeline')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('bootstrap', help='Fetch pinned public model and metadata; never original RGB or Seattle images')
    s.add_argument('--no-model', action='store_true')
    s = sub.add_parser('prepare', help='Make geographic pilot manifests from public training metadata')
    s.add_argument('--metadata', default='data/metadata/train__corrected_all_3city_remove_building.txt')
    s.add_argument('--output', default='data/manifests/pilot')
    s.add_argument('--size', type=int, default=256)
    s.add_argument('--views', type=int, default=4)
    s.add_argument('--seed', type=int, default=17)
    s = sub.add_parser('validate-data')
    s.add_argument('--manifest', default='data/manifests/pilot/all.jsonl')
    s.add_argument('--data-root', required=True)
    s.add_argument('--output', default='reports/data-readiness.json')
    s = sub.add_parser('profile')
    s.add_argument('--output', default='runs/profile128')
    s.add_argument('--size', type=int, default=128)
    s.add_argument('--chunk-rows', type=int, default=4)
    s.add_argument('--steps', type=int, default=3)
    s.add_argument('--amp', action='store_true')
    s.add_argument('--full-loss', action='store_true', help='Include LPIPS and differentiable SegFormer structural losses')
    s.add_argument('--accumulation', type=int, default=1)
    for name in ('labels', 'style'):
        s = sub.add_parser(name)
        s.add_argument('--manifest', required=True)
        s.add_argument('--data-root', required=True)
        s.add_argument('--output', required=True)
        if name == 'labels':
            s.add_argument('--revision')
    s = sub.add_parser('train')
    s.add_argument('--config', required=True)
    s.add_argument('--data-root', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--seed', type=int)
    s.add_argument('--resume', action='store_true')
    s.add_argument('--stop-after', type=int, help='Stop at this absolute optimizer step and save resumable state')
    s = sub.add_parser('generate')
    for name in ('config', 'manifest', 'data-root', 'output'):
        s.add_argument('--' + name, required=True)
    s.add_argument('--checkpoint')
    s.add_argument('--audit-freeze')
    s.add_argument('--stress', choices=['clean', 'resolution', 'blur', 'occlusion', 'yaw_error', 'position_error'], default='clean')
    s = sub.add_parser('evaluate')
    for name in ('manifest', 'predictions', 'data-root', 'output'):
        s.add_argument('--' + name, required=True)
    s.add_argument('--manual-index')
    s.add_argument('--revision')
    s.add_argument('--kid', action='store_true')
    s = sub.add_parser('freeze-audit')
    s.add_argument('--selection', required=True)
    s.add_argument('--manifest', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('select')
    s.add_argument('--candidates', required=True)
    s.add_argument('--reference-summary', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('ensemble')
    s.add_argument('--evaluations', nargs='+', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('calibrate')
    s.add_argument('--scores', required=True)
    s.add_argument('--score', choices=['disagreement', 'entropy'], default='disagreement')
    s.add_argument('--output', required=True)
    s = sub.add_parser('reliability')
    s.add_argument('--scores', required=True)
    s.add_argument('--calibration', required=True)
    s.add_argument('--output', required=True)
    s = sub.add_parser('report')
    s.add_argument('--specification', default='configs/report-pending.json')
    s.add_argument('--output', default='reports/decision')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'bootstrap':
            from .assets import bootstrap
            result = bootstrap(not args.no_model)
        elif args.command == 'prepare':
            from .manifests import prepare
            result = prepare(args.metadata, args.output, size=args.size, views=args.views, seed=args.seed)
        elif args.command == 'validate-data':
            from .manifests import validate_rgb
            result = validate_rgb(args.manifest, args.data_root, args.output)
        elif args.command == 'profile':
            from .runs import profile
            result = profile(args.output, args.size, args.chunk_rows, args.steps, args.amp, args.full_loss, args.accumulation)
        elif args.command in ('labels', 'style'):
            from .semantics import make_labels, make_style
            if args.command == 'labels':
                result = make_labels(args.manifest, args.data_root, args.output, revision=args.revision)
            else:
                result = make_style(args.manifest, args.data_root, args.output)
        elif args.command == 'train':
            from .runs import train
            result = train(args.config, args.data_root, args.output, args.seed, args.resume, args.stop_after)
        elif args.command == 'generate':
            from .runs import generate
            result = generate(args.config, args.manifest, args.data_root, args.output, args.checkpoint, args.audit_freeze, args.stress)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            result = evaluate(args.manifest, args.predictions, args.data_root, args.output, revision=args.revision,
                              manual_index=args.manual_index, kid=args.kid)
        elif args.command == 'freeze-audit':
            from .runs import freeze_audit
            result = freeze_audit(args.selection, args.manifest, args.output)
        elif args.command == 'select':
            from .evaluation import select_checkpoint
            result = select_checkpoint(args.candidates, args.reference_summary, args.output)
        elif args.command == 'ensemble':
            from .evaluation import ensemble_scores
            result = ensemble_scores(args.evaluations, args.output)
        elif args.command == 'calibrate':
            from .reliability import fit_calibration
            result = fit_calibration(read_jsonl(args.scores), score=args.score)
            save_json(args.output, result)
        elif args.command == 'reliability':
            from .reliability import reliability_metrics
            result = reliability_metrics(read_jsonl(args.scores), load_json(args.calibration))
            save_json(args.output, result)
        else:
            from .evaluation import make_report
            result = make_report(args.specification, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ResearchError, FileNotFoundError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
