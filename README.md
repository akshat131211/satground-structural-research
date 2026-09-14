# SatGround: building structure from one satellite image

A research pipeline for predicting a camera-specified ground view from one overhead RGB image. It adapts the released Sat3DGen scene features with a shared residual adapter and tests whether building-region and boundary supervision help beyond ordinary RGB/perceptual adaptation.

**Current status:** the pinned checkpoint runs on the RTX 4050 laptop. Synthetic software tests exercise the complete differentiable rendering and loss path. On 14 September 2026, the user confirmed that they obtained VIGOR access and started downloading it. **VIGOR remains the selected dataset; download completion and local validation are pending. There are no real-data adaptation results or demonstrated research improvements yet.** See [the measured status](reports/IMPLEMENTATION_STATUS.md).

We are continuing the original VIGOR + Sat3DGen structural-adapter approach. The [alternative-data review](docs/DATA_ALTERNATIVES.md) and [GroundScape access audit](docs/GROUNDSCAPE_ACCESS_AUDIT.md) are retained as background only; no dataset migration is planned.

## What is implemented

- Pinned public model and metadata downloads, checksums, isolated Python environment.
- Geographic training, validation, calibration, and audit manifests; Seattle is blocked in pilot commands.
- A zero-initialized 4,448-parameter scene adapter, frozen pretrained weights, cached features, checkpointed ray rendering, and full-image super-resolution.
- A0–A3 configurations, gradient accumulation, checkpoint/resume, training-only illumination, and pseudo-label provenance.
- Fixed-pose prediction, perturbation tests, independent-segmenter evaluation, building IoU and normalized symmetric boundary distance, LPIPS/PSNR/SSIM, optional KID, and failure galleries.
- Validation checkpoint selection, an audit freeze, geographic paired bootstrap, a conservative server gate, and calibration utilities for A4.
- CPU protocol tests, an opt-in CUDA comparison against the official renderer, and GitHub Actions tests.

Real RGB preparation, the 100-view learning check, the controlled three-seed experiments, and human annotation review remain to be performed after the download is complete and validated. External geometry and 3D expansion are conditional later phases, not completed features.

## Quick start on this laptop

Run commands from the repository root in PowerShell:

```powershell
# The environment and pinned assets have already been installed on this laptop.
.\scripts\run.ps1 --help
& .\.venv\python.exe -m pytest -q

# Safe without VIGOR: public metadata only.
.\scripts\run.ps1 prepare
.\scripts\run.ps1 validate-data --data-root data/vigor

# Real checkpoint, explicitly synthetic input and targets: resource test only.
.\scripts\run.ps1 profile --size 128 --steps 30 --full-loss --output runs/my-profile
```

On a fresh Windows machine with Conda and Git installed, run `scripts/setup.ps1`. NVIDIA CUDA hardware is required by the upstream renderer; CPU CI only runs protocol tests. Use [the runbook](docs/RUNBOOK.md) for the real-data sequence and [the data-access instructions](docs/DATA_ACCESS.md) for VIGOR.

## Experiment definitions

| Run | Trainable component | Objective |
|---|---|---|
| A0 | None | Frozen pretrained baseline |
| A1 | Shared scene adapter | RGB L1 + LPIPS |
| A2 | Same adapter | A1 + building-region loss |
| A3 | Same adapter | A2 + building-boundary loss |
| A4 | No extra image generator | Calibrated error prediction from independent A3 adapters |

The main comparison is A3 versus A1. Use the same samples, rendering settings, training budget, fixed training illumination, and seeds 17, 29, and 43. Current configurations use 256-pixel outputs after successful laptop profiling; 128 pixels remains a fallback. Change every compared configuration and regenerate manifests together when choosing a different resolution.

## Source and artifact policy

The [Sat3DGen repository](https://github.com/qianmingduowan/Sat3DGen) is checked out at `70763aff4d9fc5508e67685c906f84d3cebb34fe`. Its [public checkpoint](https://huggingface.co/qian43/Sat3DGen) is pinned to `fa0ed75aff7f42dcd749764787c0184fcab0b12c`. Third-party source, data, and model licenses remain applicable; they are not relicensed by this project.

Git tracks code, configurations, tests, documentation, and compact diagnostic reports. Original imagery, derived labels, raw prediction records, model weights, caches, credentials, and environments stay outside Git. No exact facade-recovery, metric-height, novelty, or improvement claim is established by these software checks.

Further reading: [research protocol](docs/RESEARCH_PROTOCOL.md), [related-work matrix](docs/RELATED_WORK.md), [GitHub workflow](docs/GITHUB.md).
