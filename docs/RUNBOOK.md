# Running the laptop experiments

All commands run from the project root. On this laptop use `.\scripts\run.ps1`; on a CUDA Linux server use `python -m satground` in an equivalent pinned environment. Never interpret the synthetic profile as a real-data experiment.

## 1. Environment, assets, and protocol checks

```powershell
.\scripts\setup.ps1
& .\.venv\python.exe -m pytest -q
$env:SATGROUND_GPU_TESTS = '1'
& .\.venv\python.exe -m pytest -q tests/test_backend_gpu.py
Remove-Item Env:SATGROUND_GPU_TESTS
.\scripts\run.ps1 profile --size 128 --steps 30 --full-loss --output runs/profile-new
```

`bootstrap` verifies the unmodified vendor revision, downloads the pinned safetensors checkpoint and public text metadata, and records checksums in `artifacts/asset-lock.json`. It reuses the recorded supplement revision. `profile` tests identity at zero adapter initialization, backward gradients, frozen base parameters, and a resumed optimizer step.

## 2. Obtain and verify RGB data

Follow [DATA_ACCESS.md](DATA_ACCESS.md), then:

```powershell
.\scripts\run.ps1 prepare
.\scripts\run.ps1 validate-data --data-root data/vigor
.\scripts\run.ps1 style --manifest data/manifests/pilot/train.jsonl --data-root data/vigor --output data/style/train-mean.json
.\scripts\run.ps1 labels --manifest data/manifests/pilot/train.jsonl --data-root data/vigor --output data/labels/train
```

Review the validation report before training. Resolve exact cross-split duplicates, inspect near-duplicate candidates, verify the footprint bound, and review the development alignment. The main 5,000-step study should begin only after this QA. All training labels are pseudo-labels until a human reviews them. The label index stores target and label checksums plus the frozen semantic-model revision.

For main training, complete `configs/data-qa-template.json` from actual review and save it locally as `data/qa/review.json`, including the SHA-256 of the reviewed `reports/data-readiness.json`. Training beyond step 100 requires this record. Preliminary learning checks still require complete file and exact-duplicate verification. Do not fill review booleans merely to bypass the gate.

For the planned 100-view learning check, create a deterministic subset of training views in a separate manifest, copy the A1/A3 configurations with that manifest path and a short step budget, then regenerate its style and labels. Keep those learning-check runs separate from the pilot. Do not tune against the audit set.

## 3. Baseline and controlled adaptation

```powershell
.\scripts\run.ps1 generate --config configs/A0.yaml --manifest data/manifests/pilot/validation.jsonl --data-root data/vigor --output runs/A0-validation
.\scripts\run.ps1 evaluate --manifest data/manifests/pilot/validation.jsonl --predictions runs/A0-validation --data-root data/vigor --output runs/A0-validation-eval

# Short interruption/resume check using a real training manifest.
.\scripts\run.ps1 train --config configs/A1.yaml --data-root data/vigor --output runs/A1-seed17 --stop-after 100
.\scripts\run.ps1 train --config configs/A1.yaml --data-root data/vigor --output runs/A1-seed17 --resume

# Repeat A1/A2/A3 for seeds 17, 29, 43 after the learning check and data QA.
.\scripts\run.ps1 train --config configs/A3.yaml --data-root data/vigor --output runs/A3-seed17
.\scripts\run.ps1 train --config configs/A3.yaml --data-root data/vigor --output runs/A3-seed29 --seed 29
```

Use the same data, 5,000 optimizer-step budget, accumulation of eight views, resolution, and ray counts across A1–A3. The predefined structural-weight grid is in `configs/protocol.json`; changes require copied named configurations and validation-only selection. An extension to 10,000 steps requires a recorded learning-curve diagnosis before audit access. The program does not launch expensive grids or server jobs automatically.

Each checkpoint stores the adapter, optimizer, scaler, RNG states, step, configuration, and identity hash. Resume rejects a changed configuration, data, labels, style, or pipeline source. Frozen features are cached in float32 to avoid an unmeasured precision change; budget approximately 26 MB per 320 × 320 tile feature tensor.

## 4. Select checkpoints on validation

Generate and evaluate candidate checkpoints using the validation manifest:

```powershell
.\scripts\run.ps1 generate --config configs/A3.yaml --checkpoint runs/A3-seed17/step-005000.pt --manifest data/manifests/pilot/validation.jsonl --data-root data/vigor --output runs/A3-seed17-validation
.\scripts\run.ps1 evaluate --manifest data/manifests/pilot/validation.jsonl --predictions runs/A3-seed17-validation --data-root data/vigor --output runs/A3-seed17-validation-eval
.\scripts\run.ps1 select --candidates configs/candidates.local.json --reference-summary runs/A1-seed17-validation-eval/summary.json --output runs/selection-seed17.json
```

The local candidate specification has a `candidates` array with objects containing `checkpoint` and `summary` paths. Selection minimizes group-weighted building-boundary error subject to LPIPS within 5% of the specified validation reference. Use A0 as the initial A1 reference, then matched A1 for A2/A3. Audit metrics are rejected by the selector. Retain matched selected optimizer steps for the conservative gate implemented here.

## 5. Human review and final audit

Review about 200 development views, recording visible building masks, ignored/occluded regions, pose errors, and failure categories. Prepare a separate audit review subset. No human-reviewed masks currently exist.

`evaluate --manual-index PATH` accepts a JSON object keyed by sample ID. Each entry needs `building_mask` and `valid_mask` PNG paths, `reviewed: true`, a real `reviewer`, and `reviewed_utc`. White mask pixels mean true. Optional `instance_errors` records reviewed missing, extra, and merged-building counts. Do not mark pseudo-labels as reviewed simply because they exist.

Before generating audit images, write a selection JSON with `validation_selection_complete: true` and a `runs` list. Every run has a `config` path and its selected `checkpoint` path (null for A0). Then:

```powershell
.\scripts\run.ps1 freeze-audit --selection configs/selection.local.json --manifest data/manifests/pilot/audit.jsonl --output runs/audit-freeze.json
.\scripts\run.ps1 generate --config configs/A3.yaml --checkpoint runs/A3-seed17/step-005000.pt --manifest data/manifests/pilot/audit.jsonl --data-root data/vigor --output runs/A3-seed17-audit --audit-freeze runs/audit-freeze.json
```

The freeze stores configuration, checkpoint, and audit-manifest hashes. A ledger prevents routine reruns of the same audit configuration/seed/stress setting. If a run crashes after reservation, investigate and document the failed run before resetting its ledger entry; do not use resets for tuning or favorable sample selection.

Stress conditions are `resolution`, `blur`, `occlusion`, `yaw_error`, and `position_error`, supplied with `generate --stress`. Targets retain the original camera for input-pose-error tests. These are controlled perturbations, not claims about actual sensor-error distributions.

Position stress shifts the width offset by 10 original pixels. If the positive shift would leave the tile, it shifts inward by 10 instead; the actual perturbed input camera is saved for every prediction.

## 6. Reliability and the server decision

Use the three selected A3 seeds to generate and evaluate the calibration split. Run `ensemble --evaluations DIR1 DIR2 DIR3 --output scores.jsonl`, then `calibrate --scores scores.jsonl --output calibration.json`. Repeat with `--score entropy` for the entropy baseline. Constant-confidence error is reported alongside the fitted model. A4 currently predicts the mean structural error across independent adapters; it does not create or score a new ensemble RGB image.

After the audit is frozen, produce audit ensemble scores and run `reliability --scores audit-scores.jsonl --calibration calibration.json --output reliability.json`. Calibration overlap is rejected. The failure event is boundary error > 0.03, fixed before audit evaluation. Report risk–coverage, ranking and calibration separately; a structural gain does not automatically validate uncertainty.

For the server decision, copy `configs/report-pending.json` to a local report specification. Add three comparison objects with `seed`, `control_metrics` (A1), and `proposed_metrics` (A3), referencing `metrics.jsonl` files. Complete review fields only after real review and reproducibility checks. Run `report --specification PATH`. Missing evidence yields `insufficient_evidence`, not a pass.

KID is optional with `evaluate --kid`; repeated subsets estimate sampling dispersion, not geographic confidence intervals. The core paired structural confidence interval resamples geographic blocks. FID, calibrated reliability intervals, independent metric geometry, navigable 3D, and server-scale comparisons remain later work.
