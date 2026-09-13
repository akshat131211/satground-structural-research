"""Promote compact factual diagnostics to Git; original data stay ignored."""
import shutil
import statistics
from pathlib import Path

from satground.common import load_json, save_json

root = Path(__file__).resolve().parents[1]
report = root / 'reports'
report.mkdir(exist_ok=True)
for source, destination in [
    ('runs/profile128-full-v2/profile.json', 'profile128-full.json'),
    ('runs/profile256-final/profile.json', 'profile256-full.json'),
    ('runs/profile256-accum8/profile.json', 'profile256-accum8.json'),
    ('artifacts/asset-lock.json', 'asset-lock.json'),
    ('data/manifests/pilot/split-report.json', 'data-preparation.json'),
]:
    shutil.copyfile(root / source, report / destination)
readiness = load_json(report / 'data-readiness.json')
save_json(report / 'data-readiness-summary.json', {k: readiness[k] for k in
          ['manifest_sha256', 'missing_count', 'status', 'near_duplicates_human_reviewed', 'footprint_bound_verified']})
profile = load_json(report / 'profile256-accum8.json')
verification = load_json(report / 'verification.json')
text = f'''# Implementation status — 13 September 2026

The laptop research infrastructure is implemented. Original VIGOR RGB access is pending, so no real-data A0 quality baseline, A1–A3 comparison, calibrated reliability result, or research improvement has been demonstrated.

## Verified locally

- Pinned Sat3DGen source and safetensors checkpoint downloaded and loaded on native Windows.
- NVIDIA RTX 4050 Laptop GPU; PyTorch 2.6.0+cu124 reports CUDA available.
- Zero-initialized adapter reproduces the frozen baseline exactly (maximum RGB difference 0).
- Full RGB/LPIPS/building/boundary objective passed synthetic 128- and 256-pixel tests.
- 256-pixel test: {profile['steps_requested']} optimizer steps with accumulation {profile['accumulation']}, then a resumed step; finite losses and gradients, frozen base parameters.
- Peak allocated VRAM: {profile['peak_vram_mib']:.1f} MiB; peak PyTorch reserved memory: {profile['peak_reserved_mib']:.1f} MiB. These exclude the desktop/driver's separate allocations.
- Trainable adapter parameters: {profile['trainable_parameters']:,}; frozen model parameters: {profile['frozen_parameters']:,}.
- Median warm synthetic optimizer-step time: {statistics.median(profile['step_seconds'][1:]):.2f} seconds. This is not a real-dataset or server throughput estimate; frozen features were already cached in memory.
- Protocol and CUDA reference-renderer checks: `{verification['checks'][0]['output'].splitlines()[-1]}`.
- Dependency consistency check passed. Real-data training rejects missing original images before creating a run.
- Public metadata split preparation produced 22,488 fixed-view records from 3,000 selected overhead tiles: 2,000 train, 250 validation, 250 calibration, 500 audit. RGB validation reports {readiness['missing_count']:,} missing files. Seattle imagery was not opened.

The controlled native-renderer equality check disables importance resampling to avoid differences in random-number request ordering; normal profiles retain the checkpoint's 48 coarse and 48 importance samples. Full-image super-resolution is applied after ray chunks are stitched.

## What remains

1. Obtain academic VIGOR RGB access and the three training-city sky-mask supplements.
2. Verify real image scale/footprints, duplicates, camera alignment and dates; review development building boundaries.
3. Run a real 100-view learning check, then matched A1/A2/A3 experiments with seeds 17/29/43 and validation-only selection.
4. Freeze the audit, complete independent/manual evaluation, and assess structural and reliability gates.
5. Start server, external-geometry, and 3D phases only if the preceding gates pass.

Current server decision: **insufficient evidence**. All GPU profile losses and checkpoints in this report are synthetic diagnostics, not research scores or trained useful adapters. The complete real-data pipeline has not yet been exercised end to end because the required images are absent. Optional KID has not been run on a real dataset. Human annotations have not been fabricated.

## Implementation corrections during verification

The panorama pitch sign was aligned with the pinned release and checked against reference crops. Temporary test files were moved into the workspace to avoid a pre-existing Windows shared-temp permission problem. NVIDIA's pinned SegFormer release uses legacy weight files, loaded with PyTorch's restricted weights-only loader; no safetensors file was assumed. The unmodified upstream code and checksum-pinned assets remain separate from this project's source.

See [the runbook](../docs/RUNBOOK.md), [data access](../docs/DATA_ACCESS.md), [verification records](verification.json), and [the main resource profile](profile256-accum8.json).
'''
(report / 'IMPLEMENTATION_STATUS.md').write_text(text, encoding='utf-8')
