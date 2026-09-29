# Running the bounded saturation revision

The frozen definition is [VALID_SURFACE_SATURATION_REVISION.md](VALID_SURFACE_SATURATION_REVISION.md)
and its exact budget is [the protocol](../configs/saturated-fitting200/protocol.json).
The prerequisite controlled screen and all eight zero-update GPU checks passed.
This authorizes one fresh matched A2/A3 pair on the original eight fitting views.
It does not authorize another seed, a longer run, or a server experiment.

The separate trial retains the original bundle and masks in place. Preparation
writes only three configurations and a private provenance record. It verifies
the completed prerequisite receipts, mask-derived support, current source and
offline model/reference assets before freezing them. Commit checked source
before launch; do not edit frozen dependencies during the comparison.

From the repository root in PowerShell:

```powershell
& .\.venv\python.exe scripts/prepare_saturated_fitting.py
& .\.venv\python.exe scripts/run_saturated_fitting.py --startup-one-update
# Inspect the retained first update, startup receipt, process exit and lock release.
& .\.venv\python.exe scripts/run_saturated_fitting.py --resume
```

The startup update is A2 step 1 of 200. Its retained `startup.pt` includes the
adapter, AdamW moments, disabled mixed-precision scaler and Python, NumPy,
Torch CPU and CUDA random states. Compatible resume loads these and requires
exact immediate readback before any training draw. It then performs the remaining
199 A2 updates and a fresh 200 A3 updates. Both runs use the same sample schedule.
The surface coefficient is zero for A2 and 0.5 for A3; common losses and reference
models remain identical. A completed step-200 checkpoint may be verified without
GPU restoration or another update, while retaining the earlier actual resume proof.

Inspect `runs/saturated-fitting200-seed17/status.json`, stage logs and training
JSONL files for measured progress. The runner owns a GPU lock and blocks duplicate
workers. On failure, preserve outputs and inspect the actual cause; recovery must
use unchanged compatible inputs and a verified checkpoint. Partial generation or
evaluation directories are deliberately retained for diagnosis.

Final evaluation uses only update 200 and scores the same eight training views
against frozen A0 and the fresh A2 control. Full and nonperimeter boundary errors
must both improve by at least 10% against both references. At least four views
must jointly improve both boundary measures against A2, while IoU and LPIPS stay
within the fixed quality limits. An absent empty-target stratum is reported as
absent. Geographic intervals describe fitting examples, not held-out performance.

After completion, reproduce the report under a fresh private path:

```powershell
& .\.venv\python.exe scripts/report_saturated_fitting.py `
  --run runs/saturated-fitting200-seed17 `
  --output runs/saturated-fitting200-seed17/report-reproduced
```

Inspect the private paired gallery before documenting the result. Publish only
checked aggregate evidence and code/documentation. All imagery, annotations,
per-view records, identities, coordinates and checkpoint payloads remain local.
Incomplete QA and fitting-only measurements preclude a server gate, novelty
claim or audited generalization result, regardless of the numeric outcome.
