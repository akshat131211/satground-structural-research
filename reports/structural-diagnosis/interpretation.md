# Pairing, coverage and boundary-gradient diagnosis

**Main finding: the current training validity mask removes nearly all positive
boundary supervision.** In 88 of the 92 training views with scored building
labels, no positive target-boundary pixels remain after filtering. Only 48
positive boundary pixels survive across all 100 training views. This is a
specific loss-design problem to address before increasing training duration.
It does not yet show that a corrected objective improves generated buildings.

This diagnosis keeps the single-image VIGOR + Sat3DGen research question and
the completed experiments unchanged. It checks source/camera consistency,
inspects real-image context and measures the existing objective's gradients.
No optimizer update, label revision, training extension or validation exclusion
is part of this work.

## Camera and source-image checks

All **21 reviewed development views** match their released source pairing
records. Their perspective crops match the pinned upstream implementation
exactly, with maximum pixel difference zero; rotation difference is also zero.
All accepted mask and target hashes passed verification. This rules out those
specific implementation mismatches, not incorrect source metadata or real-world
heading. [Aggregate camera audit](camera-audit.json).

All 100 training and 24 validation source rows also match the released pairing
metadata. This extends the record-consistency check to the entire current
pilot, without claiming independent geographic correctness.
[Source-record audit](source-records.json).

The original three-view context tool only supported yaw zero. A separate tool
now draws the actual target direction for the expanded cohort and shows three
other real panorama directions. Its first version normalized +180 degrees to
-180 degrees, causing small interpolation differences despite geometric
equivalence. A regression test caught this; the final version preserves the
exact recorded target yaw and checks target pixels explicitly. All 21 final
first-panel crops exactly equal the accepted targets. Earlier diagnostic
attempts are retained locally and are not used as the final context packet.

All 21 source-only diagrams were visually inspected. Nearby street and building
layouts are generally plausible. Specific unresolved issues include distant
structures near or beyond the overhead crop, vegetation and transport-structure
occlusion, building-versus-bridge taxonomy, and possible construction-date
differences. These are assistant observations, not independently certified
correspondences or proof of temporal mismatch. No example was removed and no
accepted mask was edited. [Coverage audit](coverage-audit.json).

Local source imagery and the review index are in
`data/review/structural-diagnosis-context21-v3/START_HERE.md`. Camera positions,
sample IDs and per-view observations remain local. Ray distances are measured
in source-image pixels; they do not establish metric distance or which ground
pixels are visible in the overhead crop. Independent heading, scale, capture
date, spatial-footprint and duplicate QA remain incomplete.

## Gradient diagnosis

The probe uses all 100 existing **training** views, covering 48 geographic groups,
at the zero-output adapter and saved conservative A3 steps 300 and 500. It uses
training pseudo-labels and the fixed training illumination. Reviewed validation
masks are not used in the gradient calculations. The renderer, existing loss
definitions and weights remain unchanged.

Weighted RGB, LPIPS, region and boundary components use coefficients 1, 0.1,
0.5 and 0.5. We compute their derivatives with respect to rendered RGB and all
4,448 adapter parameters. The term labelled A2 in this diagnostic is the **A2
objective at the same A3 parameters**, not the separately trained A2 checkpoint.
This isolates the instantaneous contribution of the boundary term.

The default-math smoke test stopped because summed component gradients differed
from the full objective by 0.0304%. Disabling TF32 for the diagnostic passed the
unchanged strict numerical check across the three smoke-test states. That
failed attempt remains local. Historical training settings and outputs were
not changed. The complete probe records its FP32 setting and checks gradient
summation on every example; adapter tensors must remain identical after each
state. This numerical adjustment is for measurement, not a claimed model fix.

The completed probe performed **300 finite-gradient checks in 650 seconds**,
using **682.6 MiB peak VRAM**, and took zero optimizer steps. Adapter parameters
were unchanged and the pretrained generator, segmentation model and perceptual
model stayed frozen. The largest relative gradient-summation errors were
9.91e-7 for rendered RGB and 5.93e-6 for adapter parameters, below the original
strict checks.

| A3 adapter state | Median boundary / region gradient norm | Median boundary / A2 gradient norm | Median cosine, A2 vs A3 gradients |
|---|---:|---:|---:|
| Initial | 4.92% | 4.85% | 0.999300 |
| Step 300 | 5.03% | 4.98% | 0.999292 |
| Step 500 | 4.98% | 4.87% | 0.999291 |

These are per-image medians of weighted adapter gradients. Cosine close to one
means nearly the same instantaneous direction. At the three states, the
boundary term opposes the A2 objective in 48, 43 and 47 of 100 views respectively.
Its contribution is real and finite, but small relative to the combined
objective. This is consistent with the similar A2/A3 behavior; it is not a
causal proof of the validation result. Full distributions and equal-group means
are in the [gradient report](gradients/report.md).

## Why the boundary target mostly disappears

The existing pseudo-label validity mask requires confidence of at least 0.7
and excludes dynamic classes. The boundary objective then erodes that mask by
one pixel so it does not use edges next to ignored pixels. Confidence is often
low around class transitions; the two operations remove almost every positive
building boundary from these cached labels.

| Current processing stage | Total positive target-boundary pixels, 100 training views |
|---|---:|
| Raw hard pseudo-label boundary | 163,814 |
| After pixel validity mask | 13,873 |
| After the boundary objective's eroded validity mask | 48 |

Only four views retain any positive boundary pixels. The 96 views without
positive boundaries include all eight empty-label views and **88 nonempty-label
views**. Of the 5,069,946 scored boundary-loss positions, just 48 have a positive
target. A separate CPU audit exactly reproduces the GPU probe's support counts.
Raw pseudo-label edges are not verified ground truth, so restoring all of them
indiscriminately would not be a justified correction.
[Boundary-support audit](boundary-support.json).

Where the scored target boundary is identically zero, the current boundary
loss reduces algebraically to the mean predicted morphological gradient in
the scored regions. It acts as a smoothness-like penalty there, rather than
providing positive contour-alignment supervision. A synthetic regression test
reproduces this case and verifies that equality. This is a mismatch between
the intended supervision and the available masked targets, not a detached
gradient or non-finite optimizer failure. No historical mask or loss was
silently modified to make the diagnosis look better.

## Concrete correction to investigate next

Prioritize **separate confidence handling for contours and region interiors**
before a longer training schedule. The region mask can remain conservative,
while contour supervision needs evidence for a transition from building to a
static nonbuilding class. A candidate could use soft building probabilities
and confident evidence on both sides of a transition, with dynamic/ambiguous
regions explicitly excluded. Its target support must be audited before any
training, rather than merely increasing the old boundary coefficient.

This is a proposed correction within the existing structural-supervision
question, not an implemented replacement loss or novelty claim. The next
controlled test should first verify useful positive/negative contour targets
on training examples, freeze that target definition, and compare the existing
A2 control with the revised boundary objective under a bounded matched budget.
Do not reuse the 21 reviewed validation masks for training or tune the rule on
their scores. Independent review of training-side contours is still needed
before treating them as verified labels.

## Scope of conclusions

Raw gradient size does not establish the effective AdamW update: optimizer
moments, accumulation, clipping and weight decay are outside this decomposition.
A smaller or conflicting boundary signal can motivate a specific experiment,
but does not prove that increasing its weight improves buildings. Any such
comparison must retain matched budgets and avoid selecting weights on the
already inspected validation masks.

No source pairing has been newly certified by a human in this diagnosis, and
no unknown data-QA field is marked complete. Seattle, calibration and audit
imagery remain sealed. These observations cannot support novelty, audited
generalization or a server-gate pass.

Reproduce the completed checks from the checkout root with fresh output paths:

```powershell
& .\.venv\python.exe scripts/audit_development_sources.py --output runs/source-audit-recheck.json
& .\.venv\python.exe scripts/audit_boundary_support.py --output runs/boundary-audit-recheck.json --local-output runs/boundary-audit-recheck-raw.json
& .\.venv\python.exe scripts/report_structural_gradients.py --output runs/gradient-report-recheck
```

The full gradient probe can be reproduced with
`scripts/diagnose_objective_gradients.py --output <fresh-local-directory>` when
the GPU is free. The camera/context scope is documented
in [the protocol](../../docs/STRUCTURAL_DIAGNOSIS.md). Original experiments,
checkpoints, accepted masks and sealed splits remain unchanged.
