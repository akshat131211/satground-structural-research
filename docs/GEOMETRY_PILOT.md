# Laptop field interventions and geometry-only control

Protocol recorded on 15 September 2026 before these new runs. This is a preliminary
diagnosis of the existing adapter and a controlled optimization check. It does
not implement the proposed building-instance correspondence or semantic volume
supervision. No improvement or publication claim is assumed.

## Fixed design

1. Diagnose the completed A1 and A3 seed-17 adapters on all 24 views in the
   existing `learning100/validation.jsonl`, without selecting examples by results.
2. For each camera, draw coarse rays and importance samples from frozen A0.
   Query frozen and adapted features at identical coordinates. Render four
   combinations: baseline; adapted density with baseline color features;
   baseline density with adapted color features; both adapted.
3. Preserve the pinned point integrator, fixed training-only sky, and full-image
   super-resolution. Save opacity and composited radial distance at the native
   128-pixel ray-grid resolution and RGB at 256 pixels. Radial coordinates are
   internal model units, not measured depth, building height, or geometry error.
4. Check each shared baseline against the original chunked baseline. Check exact
   opacity/radial invariance when only appearance changes. Report the difference
   between native adapted sampling and frozen-baseline proposal sampling.
5. Evaluate all four variants with the existing B1 segmenter and LPIPS. Aggregate
   by geographic group and retain empty-building and difficult cases. Pseudo
   masks do not establish ground-truth geometry.

The native A1/A3 renderer draws importance samples from its adapted density. The
new diagnostic always uses A0's proposal. Therefore the joint diagnostic is not
identical to the original generation path; the sampling difference is measured
explicitly. All hybrid comparisons are within the new shared-sampling setting.
Frozen A0 proposals can under-sample newly occupied areas; this is a limitation
of the controlled probe and is not a proposed permanent sampling algorithm.

## Matched preliminary training

| Configuration | Density query | Color-feature query | Importance proposal |
|---|---|---|---|
| GJ | Adapted features | Adapted features | Frozen A0 |
| GD | Adapted features | Frozen baseline features | Frozen A0 |

Both use the existing 100 training views and 24 validation views, seed 17, a
zero-initialized 4,448-parameter adapter, 100 AdamW steps, accumulation 8, 256-pixel
RGB, 48 coarse plus 48 fine samples, width 16, and the same A3 RGB/perceptual and
image-space region/boundary losses. The backbone, MLP, sky, and super-resolution
weights remain frozen. GD's RGB can change because density changes visibility
and integration weights, even though its color field at fixed points cannot.

Run a short stop/resume check for GD before completing its 100-step budget.
Record finite losses/gradients, base-model freezing, peak allocated CUDA memory,
elapsed time, source hash, config, data/style/checkpoint hashes, and fixed-pose
predictions. Initialization and feature-cache state affect peak/runtime; report
their scope rather than claiming a speed advantage from unmatched measurements.

GJ and GD differ only in the field routing above. The final step is evaluated
without selecting a best checkpoint against the validation targets. Compare GD
with GJ, reporting the paired geographic bootstrap and image-quality changes.
This single-seed, short run cannot pass the existing server gate. It can reveal
whether density-only optimization fits this laptop and whether restricting
updates helps or harms this particular image-space objective.

Human review remains pending. The existing readiness checks allow preliminary
runs of at most 100 steps and block the main training budget until documented
pairing, footprint-bound, and duplicate review is complete. Do not interpret
multiple short experiments as completion of that review or as the main study.
The calibration, audit, and Seattle sets are not opened in this probe.

## Commands

Run from the repository root. Use a fresh output directory for each generation.

```powershell
& .\.venv\python.exe scripts/diagnose_fields.py --config configs/learning100/A3.yaml --run runs/learning100-A3-seed17 --manifest data/manifests/learning100/validation.jsonl --data-root data/vigor --output runs/fields-A3-seed17
# Repeat with A1. Evaluate base, density, appearance, and joint directories.
.\scripts\run.ps1 evaluate --manifest data/manifests/learning100/validation.jsonl --predictions runs/fields-A3-seed17/density --data-root data/vigor --output runs/fields-A3-seed17/density-eval

.\scripts\run.ps1 train --config configs/geometry100/GD.yaml --data-root data/vigor --output runs/geometry100-GD-seed17 --stop-after 10
.\scripts\run.ps1 train --config configs/geometry100/GD.yaml --data-root data/vigor --output runs/geometry100-GD-seed17 --resume
.\scripts\run.ps1 train --config configs/geometry100/GJ.yaml --data-root data/vigor --output runs/geometry100-GJ-seed17
.\scripts\run.ps1 generate --config configs/geometry100/GD.yaml --checkpoint runs/geometry100-GD-seed17/last.pt --manifest data/manifests/learning100/validation.jsonl --data-root data/vigor --output runs/geometry100-GD-validation
# Repeat generate and evaluate for both trained variants.
```

## Next decision

Interpret changed opacity/radial maps as intervention effects, not automatically
more accurate buildings. A positive pseudo-mask result requires human and
independent geometric validation. If GD harms performance, do not buy more GPU
capacity to hide that result. The next scientific choice is between direct
semantic/identity supervision, an optimization/capacity issue, or correcting
pairing problems, based on the actual diagnostic and reviewed examples.

The broader proposed contribution and prior-art boundaries remain in
[the research assessment](RESEARCH_DIRECTION_ASSESSMENT.md). Review work uses
[the existing 200-view packet](REVIEW_GUIDE.md). Original A0/A1/A3 outputs and the
server decision criteria remain unchanged.
