# SatGround: building structure from one satellite image

A research pipeline for predicting a camera-specified ground view from one overhead RGB image. It adapts the released Sat3DGen scene features with a shared residual adapter and tests whether building-region and boundary supervision help beyond ordinary RGB/perceptual adaptation.

**Current status (16 September 2026):** the new density-only (GD) and matched joint (GJ) controls each completed 100 real-data optimizer steps on the RTX 4050. GD reduced boundary error by only **0.12% relative to GJ**, with a confidence interval spanning zero; both used about 668 MiB peak allocated CUDA memory with cached features. **The laptop can run these controls, but no meaningful structural improvement or server-scaling case is established.** See [the geometry comparison](reports/geometry-pilot/report.md), [implementation status](reports/IMPLEMENTATION_STATUS.md), and [data-ingest evidence](reports/DATA_INGEST_STATUS.json). All 8,622 pilot RGB files decode; the earlier A1/A3 check is retained [here](reports/learning100/report.md).

We are continuing the original VIGOR + Sat3DGen structural-adapter approach. The [alternative-data review](docs/DATA_ALTERNATIVES.md) and [GroundScape access audit](docs/GROUNDSCAPE_ACCESS_AUDIT.md) are retained as background only; no dataset migration is planned.

The [research-direction assessment](docs/RESEARCH_DIRECTION_ASSESSMENT.md) explains the actual DINOv3/Sat3DGen model, current evidence, newly identified overlap with prior work, and a proposed study of building identity and geometry updates. It distinguishes proposed methods from completed experiments and recommends diagnostics before new training.

The [laptop geometry probe](docs/GEOMETRY_PILOT.md) implements separate density/appearance interventions and matched GJ/GD training controls. Both A1 and A3 diagnostics passed baseline equality and field-isolation checks on all 24 views. The [codebase guide](docs/CODEBASE_GUIDE.md) explains the actual model and source files. A [24-tile multiple-view review packet](docs/IDENTITY_REVIEW_GUIDE.md) is prepared for the proposed identity-supervision stage; it contains no certified correspondences or masks.

Guided review has completed **all three practice building/validity mask pairs**.
The [three-view diagnostic](reports/reviewed-labels-3/report.md) scores the same
saved GJ/GD predictions with identical reviewed labels and reports each view
separately. View 3 has only about 7.5% building IoU for either variant and a
visible mismatch with the real buildings; this needs pairing/camera and model
error analysis. No model was retrained, and no model-improvement claim is
supported. The remaining 21 views retain pseudo-labels. Earlier
[one-view](reports/reviewed-label-probe/report.md) and
[two-view](reports/reviewed-labels-2/report.md) reports are preserved. The [source/camera audit](reports/reviewed-labels-3/failure-audit.md) matches the
released preprocessing for all three views. Full data QA and building-identity
verification remain incomplete. See the
[exact-mask workflow](docs/MASK_REVIEW_GUIDE.md) and
[aggregate review progress](reports/REVIEW_PROGRESS.json).

To correct the masks yourself, double-click **Open Mask Editor.cmd**. The local
[drawing tool](docs/DRAWING_TOOL.md) supports brushes, erasing, polygons, zoom,
undo/redo, resumable drafts and PNG export. Saved versions remain pending review.

## What is implemented

- Pinned public model and metadata downloads, checksums, isolated Python environment.
- Geographic training, validation, calibration, and audit manifests; Seattle is blocked in pilot commands.
- A zero-initialized 4,448-parameter scene adapter, frozen pretrained weights, cached features, checkpointed ray rendering, and full-image super-resolution.
- A0–A3 configurations, gradient accumulation, checkpoint/resume, training-only illumination, and pseudo-label provenance.
- Fixed-pose prediction, perturbation tests, independent-segmenter evaluation, building IoU and normalized symmetric boundary distance, LPIPS/PSNR/SSIM, optional KID, and failure galleries.
- Validation checkpoint selection, an audit freeze, geographic paired bootstrap, a conservative server gate, and calibration utilities for A4.
- CPU protocol tests, an opt-in CUDA comparison against the official renderer, and GitHub Actions tests.
- Shared-sampling density/appearance interventions, raw geometric-change exports, and separate geometry-only adaptation controls with CUDA gradient/isolation checks.

The 100-view learning check is complete; the controlled three-seed experiments and human annotation review remain. A local 200-view development review packet is prepared with exact target crops and unfilled review records; follow [the review guide](docs/REVIEW_GUIDE.md). Two approximate-hash duplicate candidates still need review; all but the three accepted guided mask pairs remain unverified proposals or pseudo-labels. External geometry and 3D expansion are conditional later phases, not completed features.

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
| GJ | Adapter changes density and color fields | A3 losses, frozen-baseline importance proposal |
| GD | Adapter changes density only; baseline color field | Same A3 losses and proposal as GJ |

The main comparison is A3 versus A1. Use the same samples, rendering settings, training budget, fixed training illumination, and seeds 17, 29, and 43. Current configurations use 256-pixel outputs after successful laptop profiling; 128 pixels remains a fallback. Change every compared configuration and regenerate manifests together when choosing a different resolution.

## Source and artifact policy

The [Sat3DGen repository](https://github.com/qianmingduowan/Sat3DGen) is checked out at `70763aff4d9fc5508e67685c906f84d3cebb34fe`. Its [public checkpoint](https://huggingface.co/qian43/Sat3DGen) is pinned to `fa0ed75aff7f42dcd749764787c0184fcab0b12c`. Third-party source, data, and model licenses remain applicable; they are not relicensed by this project.

Git tracks code, configurations, tests, documentation, and compact diagnostic reports. Original imagery, derived labels, raw prediction records, model weights, caches, credentials, and environments stay outside Git. No exact facade-recovery, metric-height, novelty, or improvement claim is established by these software checks.

Further reading: [research protocol](docs/RESEARCH_PROTOCOL.md), [related-work matrix](docs/RELATED_WORK.md), [GitHub workflow](docs/GITHUB.md).
