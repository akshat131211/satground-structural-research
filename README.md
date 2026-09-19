# SatGround: building structure from one satellite image

A research pipeline for predicting a camera-specified ground view from one overhead RGB image. It adapts the released Sat3DGen scene features with a shared residual adapter and tests whether building-region and boundary supervision help beyond ordinary RGB/perceptual adaptation.

**Autoresearch workflow imported (19 September 2026):** Karpathy's repository is
pinned locally as a reference. The project-specific [agent program](program.md)
uses bounded experiments, preserved trial branches and fixed evaluation. The
[integration guide](docs/AUTORESEARCH.md) explains the executable review-status
and historical-ledger commands. This does not start another GPU experiment;
the training review below is still the next prerequisite.

**Next step: 12-view training review prepared.** Double-click **Open Training Review.cmd**
to inspect satellite/camera context and edit the cached training building/validity
proposals. Four distinct geographic groups per city were selected using metadata
only. Use **Save & Next** through all images; uncertain and mismatched answers
are allowed. Your 21 evaluation masks stay separate. No new training has started.
See the [review guide](docs/TRAINING_REVIEW_GUIDE.md) and
[preparation status](reports/TRAINING_REVIEW_STATUS.json).

**Contour comparison completed (18 September 2026, India):** fresh A2 and corrected
A3 finished **500 steps each, seed 17**. Corrected A3's boundary error is **1.795%
lower than fresh A2** overall and **1.688% lower on the 21 reviewed views**; both
95% geographic intervals include zero. Against legacy A3 the reduction is only
**0.029%**. Structural improvement is not established, and longer training or
server scaling is not justified by this result.

The correction restored **43,127 positive contour pixels in 90 of 92 nonempty
training views**, without changing original region labels or approved masks.
These remain pseudo-labels. All 1,000 finite training rows, ten periodic checkpoints,
48 new predictions and 177 input/source pins passed verification. Reports and
local galleries reproduce. Historical A2 retraining is not bitwise reproducible;
backward-only checks demonstrate small numerical variation, while repeat inference
from the same checkpoint reproduces 24/24 images exactly. The planned Windows
restart's discarded 43 A3 updates remain preserved and separately accounted for.
The test suite passes: **130 passed, two optional CUDA tests skipped**.

Read the [completed interpretation](reports/contour500/interpretation.md),
[aggregate comparison](reports/contour500/comparison/report.md),
[completion audit](reports/contour500/completion-audit.json), and
[run status](reports/CONTOUR_RUN_STATUS.json). The [frozen protocol](docs/CONTOUR_CORRECTION.md)
and [target audit](reports/contour500/target-support.json) remain unchanged.
No further training is queued. Data QA is incomplete and sealed sets remain sealed.

**Mask batch completed:** all 18 additional building/validity pairs are accepted,
bringing the total to 21 reviewed development views. Double-click **Open Additional
Review.cmd** to inspect the accepted masks read-only. The editor supports all-image
navigation and Save & Next for future review packets. See [drawing instructions](docs/DRAWING_TOOL.md).

**Completed diagnosis before the correction:** confidence filtering removes
all positive boundary targets in **88 of 92 training views with building labels**.
Only **48 positive boundary pixels** survive across the full 100-view training
set. In views with no surviving target edge, the boundary term becomes a
smoothness-like penalty. This concerns machine-generated training labels;
your 21 approved development mask pairs are unchanged. All 300 gradient checks
completed with finite values and zero optimizer updates. Source/camera checks
passed, but independent data QA remains incomplete. Correct and verify contour
supervision before considering longer training. The correction above is now
implemented; model improvement has not been demonstrated. See the [original diagnosis](reports/structural-diagnosis/interpretation.md)
and [completion status](reports/STRUCTURAL_DIAGNOSIS_STATUS.json).

**Longer-training assessment (17 September 2026):** all 15 saved conservative
checkpoints were evaluated using the completed masks. A3's reviewed boundary
error is 1.37% worse at step 500 than at 300; the late-change intervals include
zero. A1's LPIPS worsens 3.45% over those steps. **A very long run of the current
setup is not justified.** No training extension was launched. See the
[trajectory, interpretation and next diagnostic priorities](reports/conservative-trajectory-reviewed21/interpretation.md).
All 72 regenerated final-step images exactly reproduce the earlier predictions.

**Expanded-label result (17 September 2026):** all four models were re-scored using the
completed mask batch, retaining the same 24 development views. A3's boundary
reduction versus A1 is **0.33% across all views** and **0.055% on the 21 reviewed
views**; both 95% geographic bootstrap intervals include zero. A2 explains almost
all of the IoU/perceptual recovery. **Structural improvement is not established;
server scaling is not justified.** All 96 model images, predicted segmentations
and image-quality records are unchanged. Read the [expanded-label interpretation](reports/conservative500-reviewed21/interpretation.md),
[aggregate results](reports/conservative500-reviewed21/report.md), and
[completed review status](reports/ADDITIONAL_REVIEW_STATUS.json). New labels alter
evaluation, not predictions. Three targets still use pseudo-labels, data QA is
incomplete, and no further training was launched.

**Preserved comparison with the original three reviewed masks:** the matched
500-step A1/A2/A3 revision gave 2.06% A3/A1 boundary reduction, with an interval
spanning zero and worse boundaries on the original three reviewed views. Its
[interpretation](reports/conservative500/interpretation.md),
[audit](reports/conservative500/completion-audit.json), and
[run snapshot](reports/CONSERVATIVE_RUN_STATUS.json) remain unchanged. The expanded
label evaluation above is a separate diagnostic, not a replacement primary run.

**Earlier experiment (17 September 2026, India):** the user-authorized **500-step A1/A3 exploratory comparison completed on the RTX 4050**. A3 has 40.6% lower aggregate boundary error than A1, but **79.2% of that gain comes from a 13-pixel segmentation event in one building-free target**. Boundary error worsens on all three manually reviewed views. Against the frozen A0 reference, A3 improves aggregate boundary error by 5.8%, with slightly worse LPIPS. These mixed development results do not establish structural improvement or justify server scaling. Read the [result interpretation](reports/exploratory500/interpretation.md), [unchanged primary comparison](reports/exploratory500/report.md), [completion evidence](reports/exploratory500/completion-audit.json), and [implementation status](reports/IMPLEMENTATION_STATUS.md). Data review remains incomplete. All 8,622 pilot RGB files decode; [ingest evidence](reports/DATA_INGEST_STATUS.json) is preserved.

We are continuing the original VIGOR + Sat3DGen structural-adapter approach. The [alternative-data review](docs/DATA_ALTERNATIVES.md) and [GroundScape access audit](docs/GROUNDSCAPE_ACCESS_AUDIT.md) are retained as background only; no dataset migration is planned.

**Scope confirmed:** the active question remains whether building-region and
boundary supervision improve structural fidelity beyond ordinary adaptation.
The discussed shadow-based height-estimation idea is not part of this study.
The completed A1/A3 comparison retained its fixed evaluation protocol and every validation view.

The [research-direction assessment](docs/RESEARCH_DIRECTION_ASSESSMENT.md) explains the actual DINOv3/Sat3DGen model, earlier evidence, and overlap with prior work. Its additional method proposals remain background; the active question above is unchanged.

The [laptop geometry probe](docs/GEOMETRY_PILOT.md) implements separate density/appearance interventions and matched GJ/GD training controls. Both A1 and A3 diagnostics passed baseline equality and field-isolation checks on all 24 views. The [codebase guide](docs/CODEBASE_GUIDE.md) explains the actual model and source files. A [24-tile multiple-view review packet](docs/IDENTITY_REVIEW_GUIDE.md) is prepared for the proposed identity-supervision stage; it contains no certified correspondences or masks.

Guided review has completed **all three practice building/validity mask pairs**.
The [three-view diagnostic](reports/reviewed-labels-3/report.md) scores the same
saved GJ/GD predictions with identical reviewed labels and reports each view
separately. View 3 has only about 7.5% building IoU for either variant and a
visible mismatch with the real buildings; this needs pairing/camera and model
error analysis. No model was retrained, and no model-improvement claim is
supported. That historical diagnostic retained pseudo-labels on the other 21
views; the later expanded-label result above applies to A0-A3. Earlier
[one-view](reports/reviewed-label-probe/report.md) and
[two-view](reports/reviewed-labels-2/report.md) reports are preserved. The [source/camera audit](reports/reviewed-labels-3/failure-audit.md) matches the
released preprocessing for all three views. Full data QA and building-identity
verification remain incomplete. See the
[exact-mask workflow](docs/MASK_REVIEW_GUIDE.md) and
[aggregate review progress](reports/REVIEW_PROGRESS.json).

The new [coverage-context diagnosis](reports/reviewed-labels-3/coverage-context.md)
identifies limited satellite coverage as a plausible contributor to view 3's
failure. Its four-direction diagrams follow the released camera convention;
they do not certify real-world building correspondence or introduce a new model.

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
- Training-only objective-gradient decomposition, boundary-target support audits, and source-context diagrams that preserve each reviewed target's exact camera yaw.

The 100-view learning check and one diagnosed 500-step revision are complete;
the latter does not justify automatically starting more seeds or larger runs.
Twenty-one development targets have reviewed masks, with broader data QA still
incomplete. A local 200-view packet is prepared with exact crops and unfilled
records; follow [the review guide](docs/REVIEW_GUIDE.md). Two approximate-hash
duplicate candidates still need review. External geometry and 3D expansion are
conditional later phases, not completed features.

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
